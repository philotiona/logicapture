"""Black/white/Klein-blue local interface with explicit task stages."""
from base64 import b64encode
from dataclasses import asdict
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.active import AkinatorSession, SceneGenerator, GenerationError
from src.dsl.nodes import Exists, Count, Predicate, And, Relation
from src.dsl.verbalizer import verbalize
from src.scene import GeometryTolerance, Scene
from src.scene.renderer import draw_scene, draw_page, png_bytes
from src.synthesis import search
from src.vision import load_image, rectify_page, detect_panels, extract_scene, detection_overlay
from .panels import group_board, panel_summary

BLUE = "#002FA7"
CSS = """<style>
.stApp {background:#fff;color:#000}
.block-container {max-width:1160px;padding:28px 24px 40px}
header[data-testid="stHeader"],footer {display:none}
html,body,[data-testid="stAppViewContainer"],button,input,textarea {
 font-family:Consolas,'Liberation Mono',monospace !important}
h1,h2,h3,h4 {font-family:Consolas,'Liberation Mono',monospace !important;
 color:#000 !important;font-weight:600 !important;letter-spacing:-.025em}
h2 {font-size:1.45rem !important} h3 {font-size:1.12rem !important}
p,li {font-size:.88rem;line-height:1.5}
button,input,textarea,[data-baseweb="select"]>div,
[data-testid="stVerticalBlockBorderWrapper"],details,[data-testid="stAlert"] {
 border-radius:0 !important;box-shadow:none !important}
button {font-size:.83rem !important}
button[kind="primary"] {background:#002FA7 !important;border-color:#002FA7 !important;color:white !important}
button[kind="secondary"] {background:white !important;border-color:#bbb !important;color:black !important}
[data-testid="stAlert"] {background:#f6f6f6 !important;color:black !important;border:1px solid #ddd !important;border-left:3px solid #002FA7 !important}
[data-testid="stAlert"] svg {color:#002FA7 !important}
[data-testid="stAlert"] p {color:black !important}
[data-testid="stAlert"]>div {background:#f6f6f6 !important;color:black !important}
[data-testid="stExpander"] {border-color:#ccc !important}
[data-testid="stDataFrame"] {border:1px solid #ddd}
[data-testid="stImage"] img {border-radius:0 !important;max-width:100%}
.app-header {border-bottom:2px solid #000;padding-bottom:13px;margin-bottom:15px;
display:flex;align-items:baseline;justify-content:space-between;gap:10px}
.app-name {font-size:1.5rem;font-weight:700;letter-spacing:-.05em}
.app-name:before {content:'■';font-size:17px;color:#002FA7;margin-right:12px}
.app-caption {font-size:.66rem;color:#555;letter-spacing:.04em;text-align:right}
.stats {display:flex;gap:28px;flex-wrap:wrap;border-top:1px solid #ddd;border-bottom:1px solid #ddd;padding:10px 0;margin:12px 0}
.stat {font-size:.68rem;color:#555;text-transform:uppercase}
.stat strong {display:block;font-size:1.05rem;color:#000;margin-top:3px;font-weight:500}
.rule-output {border-left:3px solid #002FA7;padding:12px 16px;margin:12px 0;background:#f5f5f5;font-size:1.1rem;line-height:1.6}
.scene-canvas {display:block;width:100%;height:340px;object-fit:contain;border:1px solid #000;background:white;margin:10px 0 14px}
.section-index {font-size:.65rem;letter-spacing:.1em;color:#002FA7;margin:18px 0 7px;font-weight:700}
.step-line {font-size:.7rem;color:#555;border-bottom:1px solid #ddd;padding:4px 0 12px;margin-bottom:14px}
.step-line strong {color:#002FA7}
.group-heading {display:flex;justify-content:space-between;border-bottom:1px solid #000;padding:0 0 7px;margin:0 0 8px;font-size:.82rem;font-weight:bold}
.group-heading span {font-weight:normal;color:#555;font-size:.7rem}
@media(max-width:640px){.block-container{padding:18px 15px 30px}.app-caption{display:none}.scene-canvas{height:285px}.stats{gap:20px}.app-name{font-size:1.3rem}}
</style>"""


def stats(items):
    st.markdown('<div class="stats">'+''.join(f'<div class="stat">{escape(str(label))}<strong>{escape(str(value))}</strong></div>' for label, value in items)+' </div>', unsafe_allow_html=True)


def section(number, name):
    st.markdown(f'<div class="section-index">{number:02d} / {escape(name.upper())}</div>', unsafe_allow_html=True)


def rule_output(rule):
    st.markdown(f'<div class="rule-output">{escape(verbalize(rule))}</div>', unsafe_allow_html=True)


def canvas(scene):
    encoded = b64encode(png_bytes(draw_scene(scene, size=420))).decode()
    st.markdown(f'<img class="scene-canvas" alt="Question drawing" src="data:image/png;base64,{encoded}">', unsafe_allow_html=True)


def generated_examples(rule, namespace, color=False):
    if st.button("Generate examples", key=f"generate_{namespace}"):
        try:
            with st.spinner("Generating two checked examples…"):
                from src.active.path_generation import PathSceneGenerator, uses_paths
                generator = PathSceneGenerator(seed=97) if uses_paths(rule) else SceneGenerator(seed=97, color=color)
                examples = [generator.sample(rule, label, 2000) for label in (True, False)]
                st.session_state[f"examples_{namespace}"] = (rule.key, examples)
        except GenerationError as error: st.warning(str(error))
    saved = st.session_state.get(f"examples_{namespace}")
    if saved and saved[0] == rule.key:
        for col, label, scene in zip(st.columns(2), (True, False), saved[1]):
            with col:
                st.caption("A / follows the rule" if label else "B / does not follow the rule")
                image = draw_scene(scene, 280)
                st.image(image, width="stretch")
                st.download_button("Download PNG", png_bytes(image), f"{'positive' if label else 'counterexample'}.png", "image/png", key=f"save_{namespace}_{label}")


def akinator():
    st.subheader("Akinator")
    st.write("Keep a visual rule in mind. Decide whether each drawing follows it.")
    start, options = st.columns([1, 2])
    with options:
        with st.expander("Session settings"):
            color = st.checkbox("Include color concepts", key="active_color")
            seed = st.number_input("Random seed", value=42, min_value=0, step=1, key="active_seed")
            st.caption("Counts 0–6; circle, triangle, quadrilateral; fill/outline; position and containment. Counted combinations include exactly four filled triangles.")
    if start.button("Start session", type="primary", width="stretch"):
        with st.spinner("Preparing the hypothesis space…"):
            st.session_state.akinator = AkinatorSession(seed=int(seed), color=color)
        st.rerun()
    session = st.session_state.get("akinator")
    if session is not None and getattr(session, "revision", 0) != 2:
        del st.session_state.akinator
        session = None
    if session is None:
        section(1, "Choose a rule")
        st.write("Start a session when you have chosen your rule. You do not need to enter it.")
        with st.expander("Supported examples", expanded=True):
            st.markdown("- Exactly four filled triangles.\n- At least four filled triangles.\n- Four objects, all filled triangles.\n- A circle exists.\n- A triangle is left of a circle.\n- Every filled object is inside an outline object.")
        return
    stats([("Answers", len(session.history)), ("Candidates", len(session.survivors)),
           ("Distinct predictions", session.equivalence_count)])
    work, log = st.columns([2.4, 1], gap="large")
    with work:
        if session.current is not None:
            if len(session.query_scenes) == 1:
                st.markdown(f"**Q{len(session.history)+1:02d} / Does this drawing follow your rule?**")
                canvas(session.scene)
            else:
                st.markdown(f"**Q{len(session.history)+1:02d} / Does ANY of these drawings follow your rule?**")
                st.caption("Each box is a separate drawing. Yes = at least one matches. No = none match.")
                st.image(group_board([draw_scene(scene) for scene in session.query_scenes],
                                     [f"TEST {i+1}" for i in range(len(session.query_scenes))], columns=2), width="stretch")
            yes, no, undo = st.columns([1, 1, 1])
            if yes.button("Yes", type="primary", width="stretch"):
                session.answer(True)
                st.rerun()
            if no.button("No", width="stretch"):
                session.answer(False)
                st.rerun()
            if undo.button("Undo", disabled=not session.history, width="stretch"):
                session.undo()
                st.rerun()
            st.caption("An empty drawing is a deliberate test. Judge it using the same rule.")
        elif not session.candidates:
            st.error("No candidate fits all answers. Undo an answer to correct it, or start a new session with a supported rule.")
            if st.button("Undo last answer"):
                session.undo()
                st.rerun()
        else:
            st.markdown("**Rule identified**" if len(session.candidates) == 1 else "**Remaining explanation**")
            rule_output(session.candidates[0])
            if len(session.candidates) > 1:
                st.write(f"{len(session.candidates)} programs still agree on the tested drawings. More tests may separate them.")
            controls = st.columns(2)
            if controls[0].button("More tests"):
                session.add_query_pool(int(seed)+len(session.scenes), color=session.color)
                st.rerun()
            if controls[1].button("Undo last answer", disabled=not session.history):
                session.undo()
                st.rerun()
            section(2, "Verify the rule")
            generated_examples(session.candidates[0], "active", session.color)
    with log:
        st.markdown("**SESSION LOG**")
        if session.history:
            st.dataframe([{"Q": f"{i+1:02d}", "Answer": "Yes" if answer.label else "No", "Remaining": answer.survivors}
                          for i, answer in enumerate(session.history)][-10:], hide_index=True, width="stretch")
        else: st.caption("No answers yet.")
        with st.expander("Candidate rules"):
            for rule in session.candidates[:5]: st.write(verbalize(rule))
        with st.expander("Search diagnostics"):
            st.caption(f"{len(session.rules)} initial programs · {len(session.scenes)} candidate drawings. Queries cover rare counted combinations and changes near their boundaries.")
            if session.history:
                st.line_chart({"Remaining": [len(session.rules)]+[a.survivors for a in session.history]}, color=BLUE)


@st.cache_data
def demo_page(choice):
    if choice == "Arrow directions / supplied drawing":
        # Original screenshot pixels in ordinary page cells. Expected labels
        # define the supplied groups; no rule is passed to perception/search.
        empty = [Scene() for _ in range(6)]
        page = draw_page((empty, empty))
        for group_index, group in enumerate(("A", "B")):
            for i in range(6):
                drawing = Image.open(Path("data/fixtures/arrow_paths")/f"{group}{i+1:02d}.png").convert("RGB")
                drawing.thumbnail((208, 208))
                x, y = 18+(2*group_index+i%2)*238, 82+(i//2)*238
                page.paste(drawing, (x+(220-drawing.width)//2, y+(220-drawing.height)//2))
        return png_bytes(page)
    path_concepts = {"Opposing arrows / generated paths": Predicate("arrow_relation", "opposed"),
                     "Open paths / closed loops": Predicate("topology", "open"),
                     "Straight paths / curved paths": Predicate("geometry", "straight"),
                     "Paths with arrows / plain paths": Predicate("has_arrows")}
    if choice in path_concepts:
        from src.active.path_generation import PathSceneGenerator
        examples = PathSceneGenerator(seed=113, topologies=("open", "closed")).balanced_examples(Exists(path_concepts[choice]), 6)
        return png_bytes(draw_page(([s for s, y in examples if y], [s for s, y in examples if not y])))
    circle, triangle = Predicate("shape", "circle"), Predicate("shape", "triangle")
    filled_triangle = Predicate("and", children=(triangle, Predicate("filled")))
    targets = {"Contains a circle": Exists(circle), "Exactly two objects": Count(2),
               "Four filled triangles": Count(4, filled_triangle),
               "Circle and two objects": And(Exists(circle), Count(2)),
               "Triangle left of circle": Relation("left_of", triangle, circle),
               "Filled objects inside outlines": Relation("inside", Predicate("filled"), Predicate("outline"), universal=True)}
    examples = SceneGenerator(seed=113).balanced_examples(targets[choice], 6)
    return png_bytes(draw_page(([s for s, y in examples if y], [s for s, y in examples if not y])))


@st.cache_resource
def cached_attribute_model():
    from src.vision.attribute_cnn import load_attribute_model
    return load_attribute_model()


def inspect_panel(saved, rule=None):
    with st.expander("Inspect one drawing"):
        names = []
        counters = {True: 0, False: 0}
        for _, label in saved["scenes"]:
            counters[label] += 1
            names.append(f"{'A' if label else 'B'}{counters[label]:02d}")
        name = st.selectbox("Drawing", names, key="inspect_panel")
        index = names.index(name)
        scene = saved["scenes"][index][0]
        evidence = rule.explain(scene) if rule else None
        left, right = st.columns([1, 1.7])
        left.image(detection_overlay(saved["images"][index][0], scene, evidence), width="stretch")
        right.dataframe([{"ID": obj.id,
                          "Primitive": "Path" if obj.kind == "path" else obj.shape if obj.shape != "unknown" else "Free-form region",
                          "Topology": obj.path.topology if obj.path and obj.kind == "path" else "Closed region",
                          "Arrows": str(len(obj.path.arrows)) if obj.path and obj.kind == "path" else "—",
                          "Along-path flow": obj.path.arrow_relation if obj.path and obj.kind == "path" else "—",
                          "Fill": "—" if obj.kind != "region" else "Filled" if obj.filled else "Outline",
                          "Source": {"stroke_graph": "Stroke graph", "neural_argmax": "CNN", "neural_rejected": "Measured / CNN rejected"}.get(obj.attribute_source, "OpenCV")}
                         for obj in scene.objects], hide_index=True, width="stretch")
        paths = [obj.path for obj in scene.objects if obj.kind == "path" and obj.path]
        if paths:
            st.caption("Arrow direction is measured along the traced path. Opposed = arrows travel opposite ways; same = arrows travel the same way, including around bends. Blue marks show the measurements.")
            with st.expander("Arrow measurements"):
                st.dataframe([{"Path": obj.id, "Arrow": i+1,
                               "Direction on traversal": "+" if arrow.direction > 0 else "−" if arrow.direction < 0 else "Unresolved",
                               "Evidence score": round(arrow.confidence, 2), "Arc position": round(arrow.arc_position, 2)}
                              for obj in scene.objects if obj.path and obj.kind == "path" for i, arrow in enumerate(obj.path.arrows)], hide_index=True, width="stretch")
                st.caption("Traversal order is arbitrary. Reversing it flips all signs together and preserves same/opposed. Evidence scores are geometric heuristics, not calibrated probabilities.")
        if rule:
            st.write(f"{name}: rule is **{'Unresolved' if evidence['matches'] is None else 'True → A' if evidence['matches'] else 'False → B'}**.")
            st.dataframe(panel_summary(rule, scene), hide_index=True, width="stretch")
        for warning in scene.warnings: st.warning(warning)
        with st.expander("Program evidence and attribute probabilities"):
            if evidence: st.json(evidence)
            st.json(scene.to_dict())


def review_groups(saved, rule=None):
    draw_objects = st.checkbox("Show detected objects", value=False, key="show_objects")
    for column, label, title in zip(st.columns(2, gap="large"), (True, False), ("GROUP A", "GROUP B")):
        indices = [i for i, (_, y) in enumerate(saved["scenes"]) if y == label]
        with column:
            st.markdown(f'<div class="group-heading">{title}<span>{len(indices)} drawings</span></div>', unsafe_allow_html=True)
            images, captions = [], []
            for number, index in enumerate(indices, start=1):
                scene = saved["scenes"][index][0]
                image = saved["images"][index][0]
                images.append(detection_overlay(image, scene, rule.explain(scene) if rule else None) if draw_objects else image)
                count = len(scene.objects)
                path_count = sum(obj.kind == "path" for obj in scene.objects)
                kind = "path" if path_count == count and count == 1 else "paths" if path_count == count and count > 1 else "object" if count == 1 else "objects"
                captions.append(f"{'A' if label else 'B'}{number:02d} / {count} {kind}")
            st.image(group_board(images, captions), width="stretch")
            if rule:
                predictions = [rule.assess(saved["scenes"][i][0]) for i in indices]
                correct = sum(value is label for value in predictions)
                unknown = sum(value is None for value in predictions)
                st.caption(f"{correct}/{len(indices)} classified correctly"+(f" · {unknown} unresolved" if unknown else ""))


def solver():
    st.subheader("Bongard solver")
    approach = st.radio("Reasoning approach", ["General visual inference", "Executable structural rules"], horizontal=True)
    if approach == "General visual inference":
        from .general_bongard import general_bongard
        general_bongard()
        return
    symbolic_solver()


def symbolic_solver():
    st.write("Find a rule that is true for Group A and false for Group B.")
    st.markdown('<div class="step-line">01 INPUT &nbsp; / &nbsp; 02 REVIEW DRAWINGS &nbsp; / &nbsp; 03 INFER RULE</div>', unsafe_allow_html=True)
    section(1, "Input")
    mode = st.radio("Source", ["Demo", "Upload page", "Separate groups", "Camera"], horizontal=True, label_visibility="collapsed")
    page, sources = None, []
    if mode == "Demo":
        choice = st.selectbox("Example problem", ["Contains a circle", "Exactly two objects", "Four filled triangles", "Circle and two objects", "Triangle left of circle", "Filled objects inside outlines", "Arrow directions / supplied drawing", "Opposing arrows / generated paths", "Open paths / closed loops", "Straight paths / curved paths", "Paths with arrows / plain paths"])
        page = demo_page(choice)
        st.caption("The solver reads this page from pixels. The intended rule is not given to the search.")
    elif mode == "Upload page":
        st.caption("Use the template layout: Group A on the left, Group B on the right.")
        uploaded = st.file_uploader("Page image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded: page = uploaded.getvalue()
        st.download_button("Blank template", Path("data/bongard_template.png").read_bytes(), "bongard_template.png", "image/png")
    elif mode == "Separate groups":
        left, right = st.columns(2)
        a = left.file_uploader("Group A / one drawing per file", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        b = right.file_uploader("Group B / one drawing per file", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        sources = [(file.getvalue(), label, file.name) for label, files in ((True, a), (False, b)) for file in files]
    else:
        captured = st.camera_input("Capture a complete template page")
        if captured: page = captured.getvalue()
    with st.expander("Extraction settings"):
        perception = st.radio("Perception pipeline", ["Structural graph + symbolic", "Structural graph + neural regions", "OpenCV + symbolic", "Neural perception + hard symbolic"])
        structural = perception.startswith("Structural")
        neural = perception.startswith("Neural") or perception == "Structural graph + neural regions"
        if structural:
            st.caption("Ink → paths and closed regions → topology and arrow measurements → executable rules. Uncertain readings remain unresolved. Shape CNNs operate only on compatible closed regions.")
            if neural:
                st.caption("The guarded CNN provides probability inspection for compatible regions. It cannot change attributes that conflict with the measured boundary; the unguarded neural baseline remains a separate comparison.")
        else:
            st.caption("Historical geometric baseline. Treats every contour as a primitive region; unsuitable for arbitrary curves and arrow-flow rules.")
        if neural:
            st.caption("CNN shape/fill distributions are retained; their most probable values feed hard symbolic reasoning.")
        a, b = st.columns(2)
        adaptive = a.checkbox("Adaptive lighting threshold")
        normalize = a.checkbox("Correct page perspective")
        remove_panel_ruling = a.checkbox("Exclude isolated panel ruling", value=mode != "Separate groups")
        st.caption("Ruling removal can also exclude a large L-shaped drawing near the edge. Disable it if that structure is part of the problem; original images remain visible.")
        margin = a.slider("Position tolerance", 0.0, .15, .05, .01)
        divider = b.slider("Divider position", .2, .8, .5, .01)
        panel_count = b.number_input("Cells per group", 1, 20, 6)
        corners_text = st.text_input("Manual page corners [x,y]", placeholder="[[20,20],[950,25],[945,770],[25,775]]")
        classify_all = a.checkbox("CNN: classify every object")
        uncertainty_threshold = b.slider("CNN: uncertainty trigger", .5, 1.0, .93, .01)
    source_key = sha256((page or b"")+b"".join(data+bytes([label]) for data, label, _ in sources)+repr((5, perception, adaptive, normalize, margin, divider, panel_count, corners_text, classify_all, uncertainty_threshold, remove_panel_ruling)).encode()).hexdigest()
    if st.button("Read drawings", type="primary", disabled=not (page or sources)):
        try:
            with st.spinner("Reading ink, paths, and closed regions…"):
                if mode == "Separate groups":
                    if {y for _, y, _ in sources} != {True, False}: raise ValueError("Add at least one drawing to each group.")
                    images = [(load_image(BytesIO(data)), label) for data, label, _ in sources]
                    overview = None
                else:
                    image = load_image(BytesIO(page))
                    corners = json.loads(corners_text) if corners_text.strip() else None
                    if normalize or corners is not None: image, _ = rectify_page(image, corners)
                    panels = detect_panels(image, divider=image.shape[1]*divider, expected_per_group=int(panel_count))
                    images = [(panel.image, panel.label) for panel in panels]
                    overview = image
                model = None
                if neural:
                    if not Path("data/models/attributes.keras").exists(): raise ValueError("The object CNN is not installed. Select OpenCV, or train the attribute model.")
                    model = cached_attribute_model()
                scenes = [(extract_scene(image, GeometryTolerance(position_margin=margin), adaptive,
                                         attribute_model=model, uncertain_only=not classify_all,
                                         uncertainty_threshold=uncertainty_threshold,
                                         representation="adaptive" if structural else "geometric",
                                         remove_panel_ruling=remove_panel_ruling), label) for image, label in images]
                st.session_state.read_drawings = {"fingerprint": source_key, "scenes": scenes, "images": images, "overview": overview}
                st.session_state.pop("solved", None)
        except (ValueError, Image.UnidentifiedImageError, json.JSONDecodeError, ImportError) as error: st.error(str(error))
    saved = st.session_state.get("read_drawings")
    if not saved or saved["fingerprint"] != source_key:
        if page:
            with st.expander("Source page preview"):
                st.image(page, width="stretch")
                if mode == "Demo": st.download_button("Download page", page, "bongard_example.png", "image/png")
        return
    section(2, "Review drawings")
    st.caption("Check the detected paths and regions before solving. Inspect a drawing to see its topology, arrow directions, and unresolved measurements.")
    with st.expander("Inference settings"):
        a, b = st.columns(2)
        color = a.checkbox("Include color predicates", key="solver_color")
        depth = a.selectbox("Composition depth", [1, 2, 3], index=1)
        noisy = b.checkbox("Allow noisy consistency scoring")
        holdout = b.checkbox("Reserve one drawing per group for evaluation", value=False)
        st.caption("Normally all drawings are used to infer the rule. Evaluation can reserve the last drawing in each group.")
    analysis_key = sha256((source_key+repr((color, depth, noisy, holdout))).encode()).hexdigest()
    analysis = st.session_state.get("solved")
    if not analysis or analysis["fingerprint"] != analysis_key: analysis = None
    ranked_best = analysis["result"].ranked[0] if analysis else None
    best_rule = ranked_best.rule if ranked_best and ranked_best.correct == ranked_best.total and ranked_best.unknown == 0 else None
    review_groups(saved, best_rule)
    inspect_panel(saved, best_rule)
    section(3, "Infer rule")
    if st.button("Infer rule", type="primary"):
        try:
            with st.spinner("Testing candidate rules…"):
                scenes, hidden = saved["scenes"], []
                if holdout:
                    if min(sum(y == label for _, y in scenes) for label in (True, False)) < 2:
                        raise ValueError("Evaluation needs at least two drawings per group. Turn off the reserve option to use all drawings.")
                    hidden = [max(i for i, (_, y) in enumerate(scenes) if y == label) for label in (True, False)]
                support = [example for i, example in enumerate(scenes) if i not in hidden]
                result = search(support, depth=depth, color=color, noisy=noisy, paths=structural)
                analysis = {**saved, "fingerprint": analysis_key, "result": result, "hidden": hidden, "perception": perception}
                st.session_state.solved = analysis
                st.rerun()
        except ValueError as error: st.error(str(error))
    if analysis is None: return
    result = analysis["result"]
    best = result.ranked[0]
    accepted = best.correct == best.total and best.unknown == 0
    if not accepted:
        st.warning("The highest ranked hypothesis is incomplete. Inspect the alternatives for complete fits." if result.perfect_count else
                   "No supported rule explains all drawings. The closest hypothesis below may reflect missing perception evidence or a concept outside the current language.")
        st.markdown("**Closest available hypothesis / incomplete explanation**")
    rule_output(best.rule)
    stats([("Support fit", f"{best.correct}/{best.total}"), ("Unresolved", best.unknown), ("Other exact fits", max(0, result.perfect_count-1)),
           ("Operations", best.rule.complexity), ("Search", f"{result.seconds:.2f}s")])
    if best.correct < best.total: st.warning("This rule does not explain all drawings. Check detected objects and group labels, or inspect the alternatives.")
    elif result.perfect_count > 1: st.info("More than one rule fits. This is the shortest ranked explanation, not a guarantee of the intended rule.")
    if analysis["hidden"]:
        values = [best.rule.assess(saved["scenes"][i][0]) for i in analysis["hidden"]]
        correct = sum(value is saved["scenes"][i][1] for value, i in zip(values, analysis["hidden"]))
        st.write(f"Reserved drawings: **{correct}/{len(analysis['hidden'])}** classified correctly; **{sum(v is None for v in values)}** unresolved.")
        if correct < len(analysis["hidden"]): st.warning("A reserved drawing was misclassified. The rule may reflect an accidental property, or perception may be wrong.")
    with st.expander("Other explanations"):
        st.dataframe([{"Rule": verbalize(r.rule), "Fit": f"{r.correct}/{r.total}", "Unresolved": r.unknown, "Operations": r.rule.complexity} for r in result.ranked], hide_index=True, width="stretch")
        if st.button("Generate a distinguishing test"):
            from src.active.path_generation import PathSceneGenerator, uses_paths
            pool = (PathSceneGenerator(818).pool(120) if any(uses_paths(r.rule) for r in result.ranked[:5]) else SceneGenerator(818, color).pool(120))
            scene = next((s for s in pool if len({r.rule.assess(s) for r in result.ranked[:5]}) > 1 and all(r.rule.assess(s) is not None for r in result.ranked[:5])), None)
            st.session_state.disagreement = (analysis_key, scene)
        disagree = st.session_state.get("disagreement")
        if disagree and disagree[0] == analysis_key:
            if disagree[1] is None: st.caption("No distinguishing drawing found in the sampled tests.")
            else:
                st.image(draw_scene(disagree[1]), width=280)
                st.dataframe([{"Rule": verbalize(r.rule), "Predicts": "A" if r.rule.assess(disagree[1]) else "B"} for r in result.ranked[:5]], hide_index=True)
    if accepted:
        section(4, "Test the rule")
        generated_examples(best.rule, "solver", color)
    with st.expander("Export and search diagnostics"):
        st.caption(f"{result.evaluated} programs tested; {result.semantic_groups} observed prediction patterns. Soft-symbolic evaluation is not implemented.")
        st.code(repr(best.rule), language="python")
        export = {"perception_pipeline": perception, "reasoning": "hard symbolic",
                  "accepted_complete_rule": accepted,
                  "rules": [{"program": asdict(r.rule), "explanation": verbalize(r.rule), "support_accuracy": r.accuracy, "coverage": r.coverage, "unresolved": r.unknown} for r in result.ranked],
                  "examples": [{"label": label, "held_out": i in analysis["hidden"], "scene": scene.to_dict()} for i, (scene, label) in enumerate(saved["scenes"])]}
        st.download_button("Download analysis JSON", json.dumps(export, indent=2), "bongard_analysis.json", "application/json")


def experiments():
    st.subheader("Experiments")
    st.caption("The supplied pilot is from the earlier grammar and query generator. It is retained as a historical run; the updated Akinator regression is evaluated separately.")
    report = Path("data/pilot/REPORT.md")
    if report.exists(): st.markdown(report.read_text(encoding="utf-8"))
    with st.expander("Architecture and limits"):
        st.write("Bongard: pixels → ink components → paths, closed regions, and unresolved entities → structural measurements → executable rules. The shape CNN is limited to compatible closed regions; its distributions are retained. Open curves and arrow flow use geometric evidence.")
        st.write("Akinator remains symbolic. The original geometric and neural pipelines remain selectable baselines. The supplied arrow drawing is one real regression example; broader hand-drawn validation and soft probabilistic evaluation remain pending. Search is bounded.")
        st.json(asdict(GeometryTolerance()))
    with st.expander("Structural regression"):
        report = Path("data/regressions/structural_regression.json")
        if report.exists(): st.json(json.loads(report.read_text(encoding="utf-8")))
    with st.expander("Object CNN validation"):
        report = Path("data/models/training.json")
        if report.exists(): st.json(json.loads(report.read_text(encoding="utf-8")))
    st.download_button("Blank drawing template", Path("data/bongard_template.png").read_bytes(), "bongard_template.png", "image/png")


def main():
    st.set_page_config(page_title="Rule Lab", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="app-header"><div class="app-name">RULE LAB</div><div class="app-caption">ACTIVE CONCEPT INFERENCE<br>BONGARD PROBLEM SOLVING</div></div>', unsafe_allow_html=True)
    if "mode" not in st.session_state or st.session_state.mode not in ("Akinator", "Bongard", "Experiments"):
        st.session_state.mode = "Akinator"
    for column, name in zip(st.columns(3), ("Akinator", "Bongard", "Experiments")):
        if column.button(name, key=f"nav_{name.lower()}", type="primary" if st.session_state.mode == name else "secondary", width="stretch"):
            st.session_state.mode = name
            st.rerun()
    if st.session_state.mode == "Akinator": akinator()
    elif st.session_state.mode == "Bongard": solver()
    else: experiments()

"""Compact local broad-concept workspace, separate from executable DSL proofs."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

from .panels import group_board


@st.cache_resource
def semantic_encoder():
    from src.vision.pretrained import SiglipEncoder
    return SiglipEncoder()


@st.cache_resource
def language_backend(model="qwen"):
    from src.vision.pretrained import QwenLocalBackend
    return QwenLocalBackend(model=model)


def _load_upload(file):
    from src.vision.pretrained import rgb_image
    with Image.open(BytesIO(file.getvalue())) as image:
        if image.width*image.height > 80_000_000: raise ValueError("Image exceeds the 80-megapixel limit.")
        if getattr(image,"n_frames",1) != 1: raise ValueError("This image is animated. A single-frame reading cannot establish its temporal rule.")
        return np.array(rgb_image(image))


def _label(value): return "A" if value is True else "B" if value is False else "Unresolved"


def general_bongard():
    from src.vision.pretrained import model_directory, image_digest, MODELS
    from src.reasoning.general import GeneralVisualSolver
    from src.vision.structural_features import StructuralEncoder

    st.write("Find a property separating A from B, then test it on unseen images.")
    st.caption("Executable drawing rules first · local visual classification for broader concepts")
    positive, negative, queries, truth = [], [], [], None
    options = ["Page image", "Live camera", "Separate images", "Supplied path example", "Supplied dotted page"]
    if Path("data/external/classic_bongard/episodes.json").is_file(): options.append("Classic problems / local data")
    if Path("data/external/foundalis/manifest.json").is_file(): options.append("Foundalis collection / local data")
    source = st.radio("Input source", options, horizontal=True, key="general_source")
    if source == "Separate images":
        left, right = st.columns(2)
        a = left.file_uploader("A / examples with the property", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="general_a")
        b = right.file_uploader("B / examples without the property", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="general_b")
        q = st.file_uploader("Unseen images / optional", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="general_q")
        try:
            positive, negative, queries = [_load_upload(f) for f in a], [_load_upload(f) for f in b], [_load_upload(f) for f in q]
        except (ValueError, Image.UnidentifiedImageError) as error: st.error(str(error)); return
        if not a and not b:
            st.info("Add four to six images to each group, or choose an example above.")
    elif source == "Page image":
        st.caption("Upload the whole page. Panel frames are found automatically; the left group becomes positive and the right group negative.")
        uploaded = st.file_uploader("Complete problem image", type=["png", "jpg", "jpeg", "webp", "gif"], key="general_page")
        with st.expander("Page layout"):
            divider = st.slider("Vertical divider", .2, .8, .5, .01, key="general_divider")
            automatic = st.checkbox("Detect panel count automatically",value=True,key="general_auto_panels")
            count = None if automatic else st.number_input("Panels per group", 2, 20, 6, key="general_panel_count")
        if uploaded:
            try:
                from src.vision.panel_detection import split_problem_page, panel_overlay
                image = _load_upload(uploaded)
                page,panels,layout = split_problem_page(image, divider=divider, expected_per_group=None if count is None else int(count))
                positive = [p.image for p in panels if p.label]
                negative = [p.image for p in panels if not p.label]
                st.caption(f"Automatically separated {len(positive)} positive and {len(negative)} negative panels. Review the crops below before solving.")
                if layout["inferred_panels"]:
                    st.info("One panel had an incomplete frame; its crop was inferred from the other eleven aligned panels. Review that boundary before solving.")
                with st.expander("Detected page boundaries"):
                    st.image(panel_overlay(page,panels),width="stretch")
            except (ValueError, Image.UnidentifiedImageError) as error: st.error(str(error)); return
    elif source == "Live camera":
        from .live_camera import live_camera_workspace
        positive,negative = live_camera_workspace()
    elif source == "Supplied dotted page":
        from src.vision.panel_detection import split_problem_page
        image = np.array(Image.open("data/fixtures/dotted_page.png").convert("RGB"))
        st.image(image,caption="Original complete page / automatically split below",width=min(700,image.shape[1]))
        _,panels,_ = split_problem_page(image)
        positive,negative = [p.image for p in panels if p.label],[p.image for p in panels if not p.label]
    elif source == "Supplied path example":
        folder = Path("data/fixtures/arrow_paths")
        positive = [np.array(Image.open(folder/f"A{i:02d}.png").convert("RGB")) for i in range(1, 7)]
        negative = [np.array(Image.open(folder/f"B{i:02d}.png").convert("RGB")) for i in range(1, 7)]
        st.caption("Arrow directions are measured along each curve. The automatic route checks executable rules before trying image features.")
    elif source == "Foundalis collection / local data":
        from src.vision.panel_detection import split_problem_page
        root = Path("data/external/foundalis")
        manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
        records = manifest["items"]
        index = st.selectbox("Problem",range(len(records)),format_func=lambda i: records[i]["id"],key="general_foundalis")
        record = records[index]
        if record["status"] != "available":
            st.warning("Source image is unavailable: "+record.get("error","retrieval failed")); return
        if len(record["images"]) != 1:
            if all(media["frames"]>1 for media in record["images"]):
                from base64 import b64encode
                st.warning("This entry uses animated panels. Temporal rule inference is not implemented; a static-frame answer would be incomplete.")
                columns = st.columns(2)
                for i,media in enumerate(record["images"]):
                    with columns[0 if i<len(record["images"])/2 else 1]:
                        data = b64encode((root/media["file"]).read_bytes()).decode()
                        st.markdown(f'<img alt="Animated problem panel" src="data:image/gif;base64,{data}" width="160">',unsafe_allow_html=True)
            else:
                st.warning("This entry contains multiple images and needs layout review; no page was silently discarded.")
            return
        item = record["images"][0]
        if item["frames"] != 1:
            st.warning("This problem uses animation; a static-frame rule would be incomplete."); return
        with Image.open(root/item["file"]) as source_image:
            from src.vision.pretrained import rgb_image
            image = np.array(rgb_image(source_image))
        st.image(image,width="stretch")
        try:
            _,panels,_ = split_problem_page(image)
            positive,negative = [p.image for p in panels if p.label],[p.image for p in panels if not p.label]
        except ValueError as error:
            st.warning(str(error)); return
        st.caption("Only the displayed image pixels and left/right groups enter inference. Source concepts and solutions are not loaded.")
    else:
        from src.datasets import load_classic, DatasetError
        episodes = load_classic("data/external/classic_bongard")
        index = st.selectbox("Problem", range(len(episodes)), format_func=lambda i: episodes[i].episode_id, key="general_classic")
        try:
            prepared = episodes[index].prepare()
            positive, negative, queries = map(list, (prepared.input.positive, prepared.input.negative, prepared.input.queries))
            truth = prepared.query_labels
        except DatasetError as error:
            st.warning(f"Strict independent evaluation excludes this problem: {error}"); return
        st.caption("Five examples per group; the sixth is hidden for evaluation. Task names and solutions do not enter inference.")
    if positive and negative and not queries:
        can_reserve = min(len(positive), len(negative)) >= 3
        reserve = st.checkbox("Withhold the last image in each group for a test", value=can_reserve,
                              disabled=not can_reserve, key="general_holdout")
        if not can_reserve: st.caption("A reserved test needs at least three images per group, leaving two supports each. You can also upload unseen images separately.")
        if reserve and min(len(positive), len(negative)) >= 3:
            # Interleave hidden queries in a deterministic pixel-based order.
            pairs = [(positive[-1], True), (negative[-1], False)]
            pairs.sort(key=lambda item: sha256(item[0].tobytes()).hexdigest())
            queries, truth = [p[0] for p in pairs], tuple(p[1] for p in pairs)
            positive, negative = positive[:-1], negative[:-1]
    if positive or negative:
        left, right = st.columns(2)
        for column, images, name in ((left, positive, "A / positive support"), (right, negative, "B / negative support")):
            with column:
                st.markdown(f"**{name}** · {len(images)} images")
                if images: st.image(group_board(images, [f"{name[0]}{i+1:02d}" for i in range(len(images))]), width="stretch")
        if queries:
            with st.expander(f"Unseen test images / {len(queries)}", expanded=True):
                st.image(group_board(queries, [f"Q{i+1:02d}" for i in range(len(queries))], tile=160), width=min(900, 172*min(3, len(queries))))
    have_semantic = model_directory("siglip") is not None
    language_models = [key for key in ("qwen8b", "qwen") if model_directory(key) is not None]
    have_language = bool(language_models)
    language_model = language_models[0] if language_models else "qwen"
    rounds = 2
    with st.expander("Local inference settings", expanded=not have_semantic):
        structural_first = st.checkbox("Try executable drawing rules first", value=True, key="general_structural_first")
        st.caption("For monochrome drawings, trace paths and regions and refit rules while withholding one support from each group. Use feature classification if this check fails. Disable to reproduce the feature-only baseline.")
        choices = ["Hybrid / learned + measured", "Learned semantic features", "Measured drawing features"] if have_semantic else ["Measured drawing features"]
        representation = st.selectbox("Representation", choices, key="general_representation")
        language = st.checkbox("Propose and check an open-language concept", value=False, disabled=not have_language, key="general_language")
        generate = st.checkbox("Draw a checked new positive and negative",value=True,key="general_generate")
        st.caption("Learned features handle natural images; measured features retain count, topology, boundaries, arrangement and symmetry. Representation selection uses support-only validation.")
        if not have_semantic or not have_language:
            st.code("python -m pip install -r requirements-general.txt\npython -m src.vision.pretrained --download siglip --download qwen", language="shell")
            st.caption("Downloads several GB of official model weights once. Inference then runs offline. No API key or usage charges.")
        if language:
            language_model = st.selectbox("Local concept model", language_models, format_func=lambda v: "Qwen3 VL 8B / NF4" if v == "qwen8b" else "Qwen3 VL 2B", key="general_language_model")
            rounds = st.slider("Concept search rounds", 1, 4, 2, key="general_concept_rounds")
            st.caption("The local model proposes concepts from support only, checks each image independently, and revises failed hypotheses. This can take several minutes. Its self-checks can still be wrong.")
    identities = tuple((key, spec.revision) for key, spec in MODELS.items())
    fingerprint = sha256(repr((4, identities, structural_first, representation, language, language_model, rounds,generate,
                               len(positive), len(negative), len(queries))).encode()+
                         "".join(image_digest(i) for i in positive+negative+queries).encode()).hexdigest()
    if st.button("Solve locally", type="primary", disabled=min(len(positive), len(negative)) < 2, key="general_solve"):
        st.session_state.pop("general_analysis", None)
        status = st.empty()
        try:
            encoder = semantic_encoder() if representation != "Measured drawing features" else None
            measured = StructuralEncoder() if representation != "Learned semantic features" else None
            result = GeneralVisualSolver(encoder, measured, structural_first=structural_first).solve(positive, negative, queries, progress=status.caption)
            saved_analysis = {"fingerprint": fingerprint, "features": result, "concepts": None, "concept_error": None,
                              "generated":None,"generation_error":None}
            st.session_state.general_analysis = saved_analysis
            if generate and result.diagnostics.get("logical_explanation"):
                from src.reasoning.examples import generate_checked_examples
                from src.active import GenerationError
                status.caption("Drawing new examples and checking their rendered pixels against the frozen programs")
                try:
                    saved_analysis["generated"] = generate_checked_examples(result.diagnostics)
                except GenerationError as error:
                    saved_analysis["generation_error"] = str(error)
            if language and not result.diagnostics.get("logical_explanation"):
                if encoder: encoder.unload()
                from src.reasoning.concepts import ConceptInducer
                status.caption("Proposing an open-language concept from support examples")
                backend = language_backend(language_model)
                try:
                    saved_analysis["concepts"] = ConceptInducer(backend, max_rounds=rounds, measured_grounding=True).solve(positive, negative, queries, progress=status.caption)
                except (ValueError, RuntimeError, ImportError, OSError) as error:
                    saved_analysis["concept_error"] = str(error)
                finally:
                    backend.unload()
        except (ValueError, RuntimeError, ImportError, OSError) as error: st.error(str(error))
        finally: status.empty()
    saved = st.session_state.get("general_analysis")
    if not saved or saved["fingerprint"] != fingerprint: return
    result, concepts = saved["features"], saved["concepts"]
    if saved.get("concept_error"):
        st.warning("The optional concept search failed; the visual decisions below are retained. " + saved["concept_error"])
    structural = result.diagnostics.get("selected_pipeline") in ("executable_structural_rules","executable_measurement_rules")
    if structural:
        st.markdown("**Inferred structural rule / group A**")
        st.write(result.diagnostics["logical_explanation"])
        attempt = result.diagnostics["structural_attempt"]
        st.caption(f"Support fit: {attempt['support_correct']}/{attempt['support_total']}. "
                   f"Leave-one-per-group validation: {attempt['validation_correct']}/{attempt['validation_total']} "
                   f"assessments across {attempt['validation_folds']} refits. Queries were not used to choose the rule.")
        st.caption(f"{attempt['candidate_count']} equally simple support-consistent programs; a query is answered only when they agree and its required evidence is available. This does not prove the intended rule is unique.")
        if attempt.get("validation_is_independent_accuracy") is False:
            st.caption("The measurement was selected using these support refits. This is a support-selection check, not independent test accuracy; withheld images below are separate.")
    else:
        st.markdown("**Visual classification / no executable rule recovered**")
        cv = result.diagnostics["nested_cv_accuracy"]
        learner = result.diagnostics["selected_candidate"]
        st.caption(f"Nested support validation: {cv:.0%}" if cv is not None else "Insufficient independent support for validation")
        st.caption(f"Selected view: {learner['view']} · learner: {learner['learner']} · scores are uncalibrated")
        if cv is not None and cv <= .5:
            st.warning("The classifier did not establish a reliable separating pattern on the supports. Its unresolved outputs are failures to solve this problem.")
        attempt = result.diagnostics.get("structural_attempt", {})
        if structural_first and attempt.get("reason"):
            st.caption("Drawing-rule check: " + attempt["reason"].replace("_", " "))
    rows = []
    for i, prediction in enumerate(result.predictions):
        row = {"Image": prediction.image_id, "Rule decision" if structural else "Feature decision": _label(prediction.label)}
        if structural and result.diagnostics["selected_pipeline"] == "executable_structural_rules":
            measured_objects = prediction.diagnostics["measured_evidence"].get("objects", [])
            row["Measured path flow"] = ", ".join(obj["along_path_flow"] for obj in measured_objects if "along_path_flow" in obj) or "—"
        elif structural:
            row["Measured values"] = ", ".join(f"{name}: {value:.3g}" if value is not None else f"{name}: unknown"
                                               for name,value in prediction.diagnostics["measured_evidence"].items())
        else:
            row["Evidence score"] = round(prediction.score, 3)
        row["Reason"] = ", ".join(prediction.abstention_reasons) or ("Executable rules agree" if structural else "Stable support fits agree")
        if concepts: row["Concept decision"] = _label(concepts.predictions[i].label)
        if truth is not None: row["Held-out truth"] = _label(truth[i])
        rows.append(row)
    if rows: st.dataframe(rows, hide_index=True, width="stretch")
    else: st.info("Add unseen images to test the decision. Support agreement alone does not establish the intended rule.")
    if truth is not None:
        from src.reasoning.evaluation import prediction_metrics
        metric = prediction_metrics(result.predictions, truth)
        st.caption(f"Held-out: {sum(p.label is y for p,y in zip(result.predictions,truth))}/{len(truth)} correct, {metric['abstained']} unresolved. Ground truth is used only here, after prediction.")
    if saved.get("generated"):
        st.markdown("**New drawings / checked against the recovered programs**")
        from src.scene.renderer import png_bytes
        for column,example in zip(st.columns(2),saved["generated"]):
            with column:
                st.image(example.image,caption="A / new positive" if example.label else "B / new negative",width="stretch")
                st.download_button("Download drawing",png_bytes(Image.fromarray(example.image)),
                    "new_positive.png" if example.label else "new_negative.png","image/png",key=f"general_example_{example.label}")
        st.caption("The drawings are generated, then re-read from pixels. Every frozen program must agree with their labels. This checks the recovered programs, not an independent proof of the intended rule.")
    elif saved.get("generation_error"):
        st.info("New example generation was not verified: "+saved["generation_error"])
    if concepts:
        st.markdown("**Open-language hypotheses**")
        for checked in concepts.candidates:
            st.write(f"A: {checked.concept.positive}")
            st.write(f"B: {checked.concept.negative}")
            st.caption(f"Group-blind checks on the same supports: {checked.support_correct}/{checked.support_total}; {checked.support_unknown} unresolved. " +
                       ("Support-consistent candidate." if checked.consistent else "Rejected as a complete explanation."))
        st.caption("A model-generated explanation is a hypothesis. Support consistency and a visual classification are separate from an executable proof.")
    with st.expander("Evidence and reproducibility"):
        for prediction in result.predictions:
            if structural:
                st.write(prediction.image_id)
                st.json(prediction.diagnostics, expanded=False)
            else:
                st.write(f"{prediction.image_id}: nearest supports {', '.join(prediction.nearest_support_ids)}")
        st.json({k: v for k, v in result.diagnostics.items() if k != "selection_cv"}, expanded=False)
        export = {"features": result.to_dict(), "concepts": concepts.to_dict() if concepts else None,
                  "concept_error": saved.get("concept_error"),
                  "generated_verification":[example.verification for example in saved.get("generated") or []],
                  "generation_error":saved.get("generation_error"),
                  "evaluation_truth": truth, "all_inference_local": True}
        st.download_button("Download analysis JSON", json.dumps(export, indent=2), "general_bongard.json", "application/json", key="general_export")

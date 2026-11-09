"""One guided Bongard workflow; model selection stays inside the solver."""
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

from src.reasoning.bongard import infer_problem
from src.vision.pretrained import image_digest, rgb_image
from .panels import group_board


def _load_upload(file):
    with Image.open(BytesIO(file.getvalue())) as image:
        if image.width*image.height > 24_000_000:
            raise ValueError("Image exceeds the 24-megapixel limit. Upload a smaller image.")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated problems are not supported. Upload a single static page.")
        return np.array(rgb_image(image))


def _example(name):
    if name == "Arrow directions":
        folder = Path(__file__).resolve().parents[2]/"data/fixtures/arrow_paths"
        return tuple([np.array(Image.open(folder/f"{letter}{i:02d}.png").convert("RGB"))
                      for i in range(1, 7)] for letter in "AB")
    from src.reasoning.examples import geometric_demo
    return tuple(map(list, geometric_demo("convexity" if name == "Convex and concave shapes" else "elongation")))


def _inputs(*, prefix="bongard", demo=None):
    source = st.radio("Problem input", ["Upload a page", "Separate images", "Try an example"],
                      horizontal=True, index=2 if demo else 0, key=f"{prefix}_source")
    if source == "Try an example":
        if demo:
            return (*demo(), False)
        name = st.selectbox("Example problem", ["Arrow directions", "Convex and concave shapes", "Long and compact shapes"], key=f"{prefix}_example")
        return (*_example(name), name == "Arrow directions")
    if source == "Separate images":
        st.caption("Upload one drawing per file. A follows the rule. B breaks it.")
        left, right = st.columns(2)
        positive = left.file_uploader("Group A drawings", type=["png", "jpg", "jpeg", "webp"],
                                      accept_multiple_files=True, key=f"{prefix}_a", max_upload_size=100)
        negative = right.file_uploader("Group B drawings", type=["png", "jpg", "jpeg", "webp"],
                                       accept_multiple_files=True, key=f"{prefix}_b", max_upload_size=100)
        framed = False
        if positive or negative:
            framed = st.checkbox("Drawings include printed panel borders", key=f"{prefix}_framed",
                                 help="Turn this on only for cropped puzzle panels with printed frames. Leave it off when rectangles are part of the drawings.")
        return ([_load_upload(file) for file in positive or []],
                [_load_upload(file) for file in negative or []], framed)
    st.caption("Upload the whole page. Logicapture separates the drawings on the left (A) from those on the right (B).")
    uploaded = st.file_uploader("Bongard problem page", type=["png", "jpg", "jpeg", "webp", "gif"],
                                key=f"{prefix}_page", max_upload_size=100)
    if not uploaded:
        st.info("Upload your problem, or choose Try an example above to see how it works.")
        return [], [], True
    image = _load_upload(uploaded)
    page_id = image_digest(image)
    if st.session_state.get(f"{prefix}_layout_image") != page_id:
        st.session_state[f"{prefix}_layout_image"] = page_id
        st.session_state[f"{prefix}_divider"] = .5
        st.session_state[f"{prefix}_count"] = "Detect automatically"
    from src.vision.panel_detection import split_problem_page, panel_overlay
    layout = st.expander("Check or fix page splitting")
    with layout:
        divider = st.slider("Boundary between groups", .2, .8, .5, .01, key=f"{prefix}_divider")
        count = st.selectbox("Drawings per group", ["Detect automatically", 3, 4, 5, 6, 7, 8], key=f"{prefix}_count")
    try:
        page, panels, _ = split_problem_page(image, divider=divider,
                                             expected_per_group=None if isinstance(count, str) else count)
    except ValueError as error:
        st.image(image, caption="Your original page", width="stretch")
        raise ValueError("The page could not be split into groups. Open Check or fix page splitting, or upload the drawings using Separate images.") from error
    layout.image(panel_overlay(page, panels), width="stretch")
    return [p.image for p in panels if p.label], [p.image for p in panels if not p.label], True


def _explain_failure(attempt):
    reason = attempt.get("reason", "")
    if "domain" in reason or "drawing" in reason:
        return "This rule finder currently needs clear black-and-white drawings. For photo categories, use Sort images."
    if "perception" in reason or "evidence" in reason:
        return "Some paths or shapes could not be read reliably. Try a sharper image with a white margin around each shape."
    return "No supported rule passed the checks across both groups. Check the groups or add varied examples of the same distinction."


def _test_images(saved):
    st.markdown("#### Test the rule")
    files = st.file_uploader("New drawings / optional", type=["png", "jpg", "jpeg", "webp"],
                            accept_multiple_files=True, key="bongard_queries", max_upload_size=100)
    if not files:
        st.caption("Optional. Add drawings that were not used to find the rule.")
        return []
    if len(files) > 24:
        st.warning("Test at most 24 drawings at a time.")
        return []
    framed = None
    if saved["result"].diagnostics.get("selected_pipeline") == "executable_structural_rules":
        framed = st.checkbox("Test drawings include printed panel borders", key="bongard_query_framed",
                             help="Leave this off when rectangles are part of the drawings.")
    # Every row shown in the table is also exported, so a report's record count
    # equals the number of attempted queries. Non-predictions carry label None.
    rows, predictions = [], []
    for file in files:
        try:
            image = _load_upload(file)
            identity = image_digest(image)
            if identity in saved["result"].model.support_digests:
                rows.append({"Image": file.name, "Result": "Given example", "Why": "This repeats a drawing used to find the rule."})
                predictions.append({"filename": file.name, "status": "given_example", "label": None,
                                    "reason": "duplicate_of_support_image", "pixel_hash": identity})
                continue
            cache = saved.setdefault("query_cache", {})
            cache_key = (identity, framed)
            if cache_key not in cache:
                cache[cache_key] = saved["result"].model.predict([image], remove_panel_ruling=framed).predictions[0]
            prediction = cache[cache_key]
            decided = prediction.label is not None
            predictions.append({"filename": file.name, "status": "predicted" if decided else "abstained",
                                "reason": "recovered_rules_agree" if decided else
                                          "; ".join(prediction.abstention_reasons) or "no_rule_consensus",
                                "pixel_hash": identity, "remove_panel_ruling": framed, **prediction.to_dict()})
            rows.append({"Image": file.name, "Result": "Group A" if prediction.label is True else "Group B" if prediction.label is False else "Cannot decide",
                         "Why": "The recovered rules agree" if prediction.label is not None else "The rules disagree or required visual evidence is missing"})
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            rows.append({"Image": file.name, "Result": "Unreadable", "Why": str(error)})
            predictions.append({"filename": file.name, "status": "unreadable", "label": None,
                                "reason": str(error)})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Test drawings use the rule you already found. To revise it, change the examples in Group A or B.")
    return predictions


def _report_payload(result, positive, negative, predictions):
    return {**result.to_dict(), "predictions": predictions,
            "support_pixel_hashes": [image_digest(image) for image in list(positive)+list(negative)]}


def _report_download(result, positive, negative, predictions=()):
    """Offer the JSON report for solved and unsolved attempts alike."""
    with st.expander("Save the explanation and evidence"):
        st.download_button("Download rule report",
                           json.dumps(_report_payload(result, positive, negative, list(predictions)),
                                      indent=2, default=str),
                           "logicapture-rule.json", "application/json", key="bongard_export")


def _evidence(saved, positive, negative):
    result = saved["result"]
    attempt = result.diagnostics["structural_attempt"]
    with st.expander("Why this rule? / evidence and alternatives"):
        st.write(f"It matches {attempt['support_correct']} of the {attempt['support_total']} given drawings and passed the internal checks with examples temporarily omitted.")
        st.caption("Agreement with given examples is not independent test accuracy. Another rule may fit them too.")
        texts = list(dict.fromkeys(item["text"] for item in attempt["candidates"]))
        if len(texts) > 1:
            st.write("Other equally simple rules that fit:")
            for text in texts[1:6]:
                st.write("• "+text)
        images = list(positive)+list(negative)
        ids = result.model.support_ids
        selected = st.selectbox("Inspect a given drawing", range(len(images)), format_func=lambda i: ids[i],
                                key="bongard_inspect_"+saved["fingerprint"][:12])
        left, right = st.columns([1, 2])
        left.image(images[selected], width=min(250, images[selected].shape[1]))
        evidence = attempt["support_evidence"][selected]
        objects = evidence.get("objects")
        if objects is not None:
            right.dataframe([{"Object": obj["id"], "Type": obj["kind"],
                              "Shape": obj["shape"],
                              "Arrow marks": obj.get("arrow_count", "—"),
                              "Direction along the path": {"opposed": "Opposite directions", "same": "Same direction"}.get(obj.get("along_path_flow"), "—")}
                             for obj in objects], hide_index=True, width="stretch")
        else:
            from src.vision.observations import DESCRIPTIONS
            used = {item["program"]["feature"] for item in attempt["candidates"]}
            right.dataframe([{"Measurement": DESCRIPTIONS.get(name, name), "Value": value}
                             for name, value in evidence.items() if name in used], hide_index=True, width="stretch")


def general_bongard():
    st.subheader("Find a visual rule")
    st.write("What do the drawings in Group A have in common that Group B does not?")
    st.markdown('<div class="workflow-step">1 · Choose your input</div>', unsafe_allow_html=True)
    try:
        positive, negative, framed = _inputs()
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        st.error(str(error))
        return
    if not positive and not negative:
        return
    fingerprint = sha256((f"bongard-product-v2:{framed}:{len(positive)}:{len(negative)}:"+
                          ":".join(image_digest(image) for image in list(positive)+list(negative))).encode()).hexdigest()
    stored = st.session_state.get("bongard_analysis")
    saved = stored if stored and stored["fingerprint"] == fingerprint else None
    if saved is None:
        st.markdown('<div class="workflow-step">2 · Check the two groups</div>', unsafe_allow_html=True)
        if stored is not None:
            st.caption("The input changed. Find the rule again to update the result.")
    with st.expander("Check the groups" if saved is None else "2 · Review the given drawings", expanded=saved is None):
        for column, images, label, description in zip(st.columns(2), (positive, negative),
                                                       ("Group A", "Group B"), ("Follows the rule", "Breaks the rule")):
            with column:
                st.markdown(f"**{label} · {description}**")
                st.caption(f"{len(images)} drawing{'s' if len(images) != 1 else ''}")
                if images:
                    st.image(group_board(images, [f"{label[-1]}{i+1:02d}" for i in range(len(images))]), width="stretch")
        st.caption("Check that each drawing is complete and in the correct group.")
    ready = 3 <= min(len(positive), len(negative)) and max(len(positive), len(negative)) <= 8
    if not ready:
        st.info(f"Add 3–8 distinct drawings per group. You have {len(positive)} in A and {len(negative)} in B.")
    if saved is None and st.button("Find the rule", type="primary", disabled=not ready, key="bongard_solve"):
        st.session_state.pop("bongard_analysis", None)
        status = st.empty()
        try:
            result = infer_problem(positive, negative, progress=status.caption, remove_panel_ruling=framed)
            saved = {"fingerprint": fingerprint, "result": result}
            st.session_state.bongard_analysis = saved
            st.rerun()
        except (ValueError, RuntimeError, OSError) as error:
            saved = None
            st.error(str(error))
        finally:
            status.empty()
    if saved is None:
        return
    result = saved["result"]
    st.markdown('<div class="workflow-step">3 · Review the result</div>', unsafe_allow_html=True)
    if result.model is None:
        st.warning("No verified rule found for this problem.")
        st.write(_explain_failure(result.diagnostics["structural_attempt"]))
        _report_download(result, positive, negative)
        return
    st.markdown('<div class="rule-output">'+escape(result.diagnostics["logical_explanation"])+"</div>", unsafe_allow_html=True)
    st.caption(f"Fits all {len(positive) + len(negative)} example drawings. Test it on new drawings to see where it holds.")
    if result.examples:
        from src.scene.renderer import png_bytes
        for column, example in zip(st.columns(2), result.examples):
            with column:
                st.write("New Group A example" if example.label else "New Group B counterexample")
                st.image(example.image, width=min(320, example.image.shape[1]))
                st.download_button("Save Group A drawing" if example.label else "Save Group B drawing",
                                   png_bytes(Image.fromarray(example.image)), "group-a.png" if example.label else "group-b.png",
                                   "image/png", key=f"bongard_drawing_{example.label}")
        st.caption("These are new drawings. Their rendered pixels were checked against every retained rule.")
    else:
        st.info("A rule was found, but new drawings could not be verified for it.")
    _evidence(saved, positive, negative)
    predictions = _test_images(saved)
    _report_download(result, positive, negative, predictions)

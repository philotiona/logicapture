"""Local Lab workspace for learned visual contrasts."""
from hashlib import sha256

from PIL import Image
import streamlit as st

from src.reasoning.local_lab import lab_demo, learn_and_draw
from src.scene.renderer import png_bytes
from src.vision.pretrained import image_digest
from .general_bongard import _inputs, _load_upload
from .panels import group_board


def _show_references(positive, negative, *, expanded):
    label = "Check Group A and Group B" if expanded else "Review the reference drawings"
    with st.expander(label, expanded=expanded):
        for column, group_label, group in zip(st.columns(2), "AB", (positive, negative)):
            with column:
                st.markdown(f"**Group {group_label} · {len(group)} drawings**")
                if group:
                    st.image(group_board(group, [f"{group_label}{i+1:02}" for i in range(len(group))]),
                             width="stretch")


def local_lab():
    st.subheader("Local Lab")
    st.write("Learn what separates A from B, then try making a new drawing that fits A.")
    st.caption("Runs on this computer using Logicapture’s own visual measurements and classifier.")
    try:
        positive, negative, framed = _inputs(prefix="lab", demo=lab_demo)
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        st.error(str(error))
        return
    if not positive and not negative:
        return

    fingerprint = sha256(f"{framed}:{len(positive)}:{len(negative)}:".encode())
    for panel in positive+negative:
        fingerprint.update(image_digest(panel).encode())
    key = fingerprint.hexdigest()
    saved = st.session_state.get("lab_result")
    result = saved[1] if saved and saved[0] == key else None
    ready = all(3 <= len(group) <= 8 for group in (positive, negative))
    if not ready:
        st.info("Add 3–8 different drawings per group.")
    if result is None and st.button("Learn and make a drawing", type="primary", disabled=not ready, key="lab_generate"):
        status = st.empty()
        try:
            result = learn_and_draw(positive, negative, framed=framed, progress=status.caption)
            st.session_state.lab_result = (key, result)
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))
        finally:
            status.empty()
    if result is None:
        _show_references(positive, negative, expanded=True)
        return

    st.markdown("#### Result")
    diagnostics = result.fit.diagnostics
    score = diagnostics["nested_cv_accuracy"]
    family = diagnostics["selected_candidate"]["view"]
    st.caption(f"Reference cross-validation: {score:.0%} balanced accuracy. "
               f"Selected measurements: {family}. Candidates checked: {result.tried}.")
    if result.candidate is None:
        st.warning(result.reason)
    else:
        st.success("This drawing fits the learned Group A contrast.")
        st.image(result.candidate, width=256)
        st.caption(f"Combined whole parts from {result.donors[0]} and {result.donors[1]}. "
                   f"Most similar reference silhouette: {result.similarity:.0%} overlap after alignment.")
        st.download_button("Save the new drawing", png_bytes(Image.fromarray(result.candidate)),
                           "logicapture-lab.png", "image/png", key="lab_save")
    st.caption("Reference checks and a candidate selected by this model give evidence about the learned contrast. "
               "Try independent drawings to see how well it generalizes.")
    _show_references(positive, negative, expanded=False)

    st.markdown("#### Test an independent drawing")
    upload = st.file_uploader("Test drawing", type=["png", "jpg", "jpeg", "webp"], key="lab_test")
    if upload is not None:
        framed_test = st.checkbox("The test drawing has a printed panel border", key="lab_test_framed")
        try:
            prediction = result.check(_load_upload(upload), framed=framed_test)
            if prediction.label is None:
                st.warning("Unclear. The drawing did not pass the confidence checks.")
            else:
                st.info("Looks like Group A." if prediction.label else "Looks like Group B.")
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            st.error(str(error))

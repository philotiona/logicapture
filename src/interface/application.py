"""Logicapture's rule discovery, drawing, image sorting and coursework."""
import streamlit as st

from .styles import CSS


def main():
    st.set_page_config(page_title="Logicapture", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="app-header"><div class="app-name">Logicapture</div>'
                '<div class="app-caption">Learn from visual contrasts and test new drawings.</div></div>',
                unsafe_allow_html=True)
    modes = ("Bongard", "Local Lab", "Sort images", "РГР coursework")
    old_mode = st.session_state.get("mode")
    aliases = {"Image tasks": "Sort images", "РГР": "РГР coursework"}
    st.session_state.mode = aliases.get(old_mode, old_mode) if old_mode in (*modes, *aliases) else "Bongard"
    # Retired game sessions can contain a large hypothesis space. They no
    # longer have a product route or controls and should not linger in memory.
    for key in list(st.session_state):
        if key == "akinator" or key.startswith(("active_", "akinator_", "examples_akinator")):
            st.session_state.pop(key, None)
    labels = ("Find a rule", "Local Lab", "Sort images", "РГР coursework")
    help_text = ("Find and check an executable rule.",
                 "Learn a local contrast and create a new drawing.",
                 "Teach two categories and sort a batch of images.",
                 "Run the separate model-testing coursework.")
    for column, name, label, key, tip in zip(st.columns(len(modes)), modes, labels,
                                           ("bongard", "lab", "image_tasks", "rgr"), help_text):
        if column.button(label, key=f"nav_{key}", width="stretch", help=tip,
                         type="primary" if st.session_state.mode == name else "secondary"):
            st.session_state.mode = name
            st.rerun()
    if st.session_state.mode == "Bongard":
        from .general_bongard import general_bongard
        general_bongard()
    elif st.session_state.mode == "Local Lab":
        from .local_lab import local_lab
        local_lab()
    elif st.session_state.mode == "Sort images":
        from .image_tasks import image_tasks_workspace
        image_tasks_workspace()
    else:
        from .rgr import rgr_workspace
        rgr_workspace()

"""Optional local resources for image sorting; Bongard needs no model download."""
import streamlit as st


@st.cache_resource
def semantic_encoder():
    from src.vision.pretrained import SiglipEncoder
    return SiglipEncoder()

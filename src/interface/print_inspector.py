"""Print Inspector workspace and printable examples."""
from io import BytesIO
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from src.scene.renderer import png_bytes
from src.vision.page_rectification import load_image
from src.vision.print_inspection import annotate, demo_card, inspect_card, learn_reference, RetakePhoto


SAMPLES = {
    "Good card": "good",
    "Missing mark": "missing",
    "Extra mark": "extra",
    "Misplaced mark": "misplaced",
}


def demo_photo(kind, *, seed=7):
    """Make a simulated photo for a demo without a printer or camera."""
    rng = np.random.default_rng(seed)
    card = np.array(demo_card(kind))
    h, w = card.shape[:2]
    y, x = np.mgrid[:h, :w]
    light = .92 + .06*x/w - .13*y/h
    paper = np.uint8(np.clip(card*light[:, :, None] + np.array([6, 3, 0]), 0, 255))
    origin = np.float32(((0, 0), (w-1, 0), (w-1, h-1), (0, h-1)))
    corners = np.float32(((156, 113), (1097, 64), (1125, 735), (119, 785))) + rng.integers(-27, 28, (4, 2))
    transform = cv2.getPerspectiveTransform(origin, np.float32(corners))
    size = (1240, 850)
    photo = cv2.warpPerspective(paper, transform, size, borderValue=(77, 87, 99))
    return cv2.GaussianBlur(photo, (5, 5), .85)


@lru_cache(maxsize=1)
def printable_pack():
    pages = []
    cases = (*SAMPLES.items(), ("Blank template", "blank"))
    for index, (title, kind) in enumerate(cases, 1):
        page = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(page)
        ink, muted = "#142844", "#41536b"
        title_font = ImageFont.load_default(size=54)
        body_font = ImageFont.load_default(size=27)
        small_font = ImageFont.load_default(size=22)
        draw.text((80, 88), "LOGICAPTURE / PRINT INSPECTOR", font=small_font, fill=muted)
        draw.text((80, 145), title, font=title_font, fill=ink)
        draw.line((80, 235, 1160, 235), fill="#c5cdd9", width=2)
        page.paste(demo_card(kind), (80, 285))
        instructions = (
            ("Draw separate dark marks in the light-gray rectangle.", "Photograph this accepted layout to teach a custom reference.")
            if kind == "blank" else
            ("Keep this sheet flat and all four corner markers visible.", "In Print Inspector, choose Upload a photo or Use camera.")
        )
        for line, instruction in enumerate(instructions):
            draw.text((80, 1080+44*line), instruction, font=body_font, fill=muted)
        draw.line((80, 1590, 1160, 1590), fill="#c5cdd9", width=2)
        draw.text((80, 1625), "Print on A4 / Keep all corner markers on the page.", font=small_font, fill=muted)
        draw.text((1160, 1625), f"{index:02d} / {len(cases):02d}", anchor="ra", font=small_font, fill=muted)
        pages.append(page)
    buffer = BytesIO()
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:], dpi=(150, 150), quality=95)
    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=12)
def _inspect(photo, reference):
    return inspect_card(photo, learn_reference(reference))


def print_inspector():
    st.subheader("Print Inspector")
    st.write("Check a printed card from a photo. Logicapture aligns the four corner markers and shows where the print differs from an accepted card.")
    with st.expander("Print your own cards"):
        st.download_button("Download the A4 demo pack", printable_pack(), "logicapture-print-inspection.pdf",
                           "application/pdf", key="inspection_print")
        st.caption("The pack includes one good card, three defect cards and a blank template for your own layout.")
    source = st.radio("Photo source", ["Try a sample", "Upload a photo", "Use camera"],
                      horizontal=True, key="inspection_source")
    reference = np.array(demo_card())
    if source == "Try a sample":
        choice = st.selectbox("Sample card", tuple(SAMPLES), key="inspection_sample")
        photo = demo_photo(SAMPLES[choice])
        st.caption("Simulated camera photo with perspective and uneven lighting.")
    else:
        st.caption("Keep the sheet flat and all four markers visible. Use dark marks about the size shown in the demo.")
        custom = st.checkbox("Teach from my good card", key="inspection_custom")
        if custom:
            accepted = st.file_uploader("Accepted reference photo", type=["png", "jpg", "jpeg", "webp"],
                                        key="inspection_reference")
            if accepted is None:
                st.info("Upload a good card made with the blank template. It defines the marks to expect.")
                return
            try:
                reference = load_image(accepted)
                learn_reference(reference)
            except (ValueError, OSError, Image.DecompressionBombError) as error:
                st.error(f"Reference photo needs attention. {error}")
                return
        upload = (st.camera_input("Take a photo of the card", key="inspection_camera") if source == "Use camera" else
                  st.file_uploader("Card photo to inspect", type=["png", "jpg", "jpeg", "webp"], key="inspection_upload"))
        if upload is None:
            return
        try:
            photo = load_image(upload)
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            st.error(str(error))
            return
    try:
        result = _inspect(photo, reference)
    except RetakePhoto as error:
        st.warning(f"Retake needed. {error}")
        st.image(photo, caption="Photo received", width="stretch")
        return
    if result.passed:
        st.success("PASS · The marks match the accepted card.")
    else:
        st.error("FAIL · The highlighted marks need review.")
    expected, observed, flagged = st.columns(3)
    expected.metric("Expected marks", result.expected)
    observed.metric("Detected marks", result.observed)
    flagged.metric("Flagged locations", len(result.defects))
    left, right = st.columns(2)
    left.image(photo, caption="Input photo", width="stretch")
    overlay = annotate(result)
    right.image(overlay, caption="Aligned card with inspection overlay", width="stretch")
    if result.defects:
        st.table([{"Issue": item.kind.capitalize(),
                   "Location": "Upper half" if (item.observed or item.expected).center[1] < 360 else "Lower half"}
                  for item in result.defects])
    st.download_button("Save inspection overlay", png_bytes(overlay), "logicapture-inspection.png",
                       "image/png", key="inspection_export")

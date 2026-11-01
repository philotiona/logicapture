"""Browser camera transport; image processing and inference stay local."""
from hashlib import sha256
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import streamlit as st


def _clear_capture():
    st.session_state.pop("general_camera_capture", None)
    st.session_state.pop("general_camera_capture_provenance", None)


def _store_capture(panels, source, token=None):
    # Freeze the actual crops from the one engine snapshot being displayed.
    # Worker updates must not replace the page the user explicitly selected.
    positive = [panel.image.copy() for panel in panels if panel.label]
    negative = [panel.image.copy() for panel in panels if not panel.label]
    if token is None:
        digest = sha256()
        for label, images in (("A", positive), ("B", negative)):
            for image in images:
                digest.update(f"{label}:{image.shape}:{image.dtype}:".encode())
                digest.update(image.tobytes())
        token = digest.hexdigest()
    st.session_state.general_camera_capture = (positive, negative)
    st.session_state.general_camera_capture_provenance = {"source": source, "token": token}
    st.session_state.pop("general_camera_snapshot_error", None)


def _snapshot_input(captured):
    """Apply a changed snapshot once, keeping an explicitly selected live page.

    The widget can retain an older photograph during live capture. Its unchanged
    bytes must not overwrite that newer selection on the ensuing full rerun.
    """
    if captured is None:
        st.session_state.general_camera_snapshot_token = None
        st.session_state.pop("general_camera_snapshot_error", None)
        provenance = st.session_state.get("general_camera_capture_provenance", {})
        if provenance.get("source") != "live":
            _clear_capture()
        return
    token = None
    try:
        data = captured.getvalue()
        token = sha256(data).hexdigest()
        if token != st.session_state.get("general_camera_snapshot_token"):
            st.session_state.general_camera_snapshot_token = token
            # A new requested input invalidates the former selection before
            # decoding or detection, including when the replacement is invalid.
            _clear_capture()
            st.session_state.pop("general_camera_snapshot_error", None)
            from src.vision.panel_detection import split_problem_page
            from src.vision.pretrained import rgb_image
            with Image.open(BytesIO(data)) as image:
                if image.width*image.height > 80_000_000:
                    raise ValueError("Image exceeds the 80-megapixel limit.")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Use a single camera frame; animated images are unsupported.")
                pixels = np.array(rgb_image(image))
            _, panels, _ = split_problem_page(pixels)
            _store_capture(panels, "snapshot", token)
    except (ValueError, OSError, TypeError, Image.DecompressionBombError, cv2.error) as error:
        _clear_capture()
        if token is None:
            st.session_state.general_camera_snapshot_token = None
        st.session_state.general_camera_snapshot_error = "Could not read the camera snapshot: "+str(error)
    if st.session_state.get("general_camera_snapshot_error"):
        st.warning(st.session_state.general_camera_snapshot_error)


def live_camera_workspace():
    st.write("Show the complete problem to the camera: positives on the left, negatives on the right.")
    st.caption("Panel outlines, a support-checked rule and pixel-checked new examples appear in the live video. Audio is disabled. No hosted inference or relay servers are configured.")
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        import av
        from src.vision.live import LivePageEngine
        if "bongard_live_engine" not in st.session_state:
            st.session_state.bongard_live_engine = LivePageEngine()
        engine = st.session_state.bongard_live_engine
        mirror = st.checkbox("Mirror camera before analysis",value=False,key="live_mirror")
        def callback(frame):
            pixels = frame.to_ndarray(format="rgb24")
            if mirror:
                pixels = np.ascontiguousarray(pixels[:,::-1])
            return av.VideoFrame.from_ndarray(engine.process(pixels),format="rgb24")
        ctx = webrtc_streamer(key="bongard_live_video",mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers":[]}, server_rtc_configuration={"iceServers":[]},
            media_stream_constraints={"video":{"width":{"ideal":1280},"height":{"ideal":720}},"audio":False},
            video_frame_callback=callback,on_video_ended=engine.reset,async_processing=True,
            video_html_attrs={"autoPlay":True,"controls":False,"muted":True})
        @st.fragment(run_every=1. if ctx.state.playing else None)
        def evidence():
            state = engine.snapshot()
            st.caption(state["message"])
            result = state["result"]
            if result is not None:
                with st.expander("Live rule evidence"):
                    st.json({k:v for k,v in result.items() if k not in ("examples","positive","negative")},expanded=False)
            if st.button("Use current camera page for full analysis", disabled=not state["panels"], key="live_capture"):
                if state["panels"]:
                    _store_capture(state["panels"], "live")
                    # A fragment-only rerun cannot refresh support groups in
                    # the parent workspace. Re-enter the complete application.
                    st.rerun(scope="app")
        evidence()
    except ImportError:
        st.info("Install requirements-camera.txt to enable live video. A single-frame capture is available below.")
    with st.expander("Single-frame camera capture"):
        captured = st.camera_input("Capture the complete problem page",key="general_camera_snapshot")
        _snapshot_input(captured)
    provenance = st.session_state.get("general_camera_capture_provenance", {})
    if provenance.get("source") == "live":
        st.caption("Full analysis uses the page selected from the live camera.")
    return st.session_state.get("general_camera_capture",([],[]))

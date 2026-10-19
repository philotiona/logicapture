"""Browser camera transport; image processing and inference stay local."""
from io import BytesIO
import numpy as np
from PIL import Image
import streamlit as st


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
        evidence()
        if st.button("Use current camera page for full analysis",disabled=not engine.snapshot()["panels"],key="live_capture"):
            panels = engine.snapshot()["panels"]
            st.session_state.general_camera_capture = ([p.image for p in panels if p.label],[p.image for p in panels if not p.label])
    except ImportError:
        st.info("Install requirements-camera.txt to enable live video. A single-frame capture is available below.")
    with st.expander("Single-frame camera capture"):
        captured = st.camera_input("Capture the complete problem page",key="general_camera_snapshot")
        if captured:
            from src.vision.panel_detection import split_problem_page
            from src.vision.pretrained import rgb_image
            with Image.open(BytesIO(captured.getvalue())) as image:
                pixels = np.array(rgb_image(image))
            try:
                _,panels,_ = split_problem_page(pixels)
                st.session_state.general_camera_capture = ([p.image for p in panels if p.label],[p.image for p in panels if not p.label])
            except ValueError as error:
                st.warning(str(error))
    return st.session_state.get("general_camera_capture",([],[]))

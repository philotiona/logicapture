"""Camera selection state without a webcam, browser, or network transport."""
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.interface import live_camera as interface
from src.vision import live, panel_detection


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class AppRerun(Exception):
    pass


class Streamlit:
    def __init__(self):
        self.session_state = State()
        self.captured = None
        self.fragments = []
        self.in_fragment = False
        self.buttons = []
        self.click = False
        self.warnings = []
        self.reruns = []
        self.periods = []

    def write(self, *args, **kwargs): pass
    def caption(self, *args, **kwargs): pass
    def json(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, message): self.warnings.append(message)
    def checkbox(self, *args, **kwargs): return False
    def expander(self, *args, **kwargs): return nullcontext()
    def camera_input(self, *args, **kwargs): return self.captured

    def fragment(self, *, run_every):
        self.periods.append(run_every)
        def decorate(function):
            def run():
                self.in_fragment = True
                try:
                    return function()
                finally:
                    self.in_fragment = False
            self.fragments.append(run)
            return run
        return decorate

    def button(self, label, *, disabled, key):
        self.buttons.append({"disabled": disabled, "inside_fragment": self.in_fragment, "key": key})
        clicked, self.click = self.click, False
        return clicked and not disabled

    def rerun(self, *, scope):
        self.reruns.append(scope)
        raise AppRerun()


class Engine:
    def __init__(self):
        self.panels = ()
        self.calls = 0
        self.after_snapshot = None

    def snapshot(self):
        self.calls += 1
        state = {"panels": self.panels, "result": None, "message": "Detected page", "epoch": 1}
        if self.after_snapshot:
            self.after_snapshot()
        return state

    def reset(self): self.panels = ()


def _panels(value=30):
    return (SimpleNamespace(image=np.full((30, 30, 3), value, np.uint8), label=True),
            SimpleNamespace(image=np.full((30, 30, 3), 230, np.uint8), label=False))


def _snapshot(value=60):
    stream = BytesIO()
    Image.new("RGB", (40, 40), (value, value, value)).save(stream, format="PNG")
    return stream


@pytest.fixture
def camera(monkeypatch):
    st, engine = Streamlit(), Engine()
    monkeypatch.setattr(interface, "st", st)
    monkeypatch.setattr(live, "LivePageEngine", lambda: engine)
    monkeypatch.setitem(sys.modules, "streamlit_webrtc", SimpleNamespace(
        webrtc_streamer=lambda **kwargs: SimpleNamespace(state=SimpleNamespace(playing=True)),
        WebRtcMode=SimpleNamespace(SENDRECV="sendrecv")))
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace())
    return st, engine


def test_periodic_fragment_enables_and_disables_capture_from_current_panels(camera):
    st, engine = camera
    assert interface.live_camera_workspace() == ([], [])
    assert st.periods == [1.]
    assert st.buttons[-1]["disabled"] and st.buttons[-1]["inside_fragment"]
    engine.panels = _panels()
    st.fragments[-1]()
    assert not st.buttons[-1]["disabled"] and st.buttons[-1]["inside_fragment"]
    assert "general_camera_capture" not in st.session_state
    engine.panels = ()
    st.fragments[-1]()
    assert st.buttons[-1]["disabled"]


def test_live_capture_uses_one_snapshot_and_freezes_crops_before_full_app_rerun(camera):
    st, engine = camera
    engine.panels = panels = _panels()
    # A frame/page can disappear immediately after the snapshot. The capture
    # must use its displayed state rather than making a second, racy read.
    engine.after_snapshot = lambda: setattr(engine, "panels", ())
    st.click = True
    with pytest.raises(AppRerun):
        interface.live_camera_workspace()
    assert engine.calls == 1 and st.reruns == ["app"]
    selected = st.session_state.general_camera_capture
    assert [len(group) for group in selected] == [1, 1]
    np.testing.assert_array_equal(selected[0][0], panels[0].image)
    assert st.session_state.general_camera_capture_provenance["source"] == "live"
    assert st.session_state.general_camera_capture_provenance["token"]
    panels[0].image[:] = 0
    assert np.all(selected[0][0] == 30)


def test_new_invalid_snapshot_invalidates_previous_page_and_retains_warning(camera, monkeypatch):
    st, _ = camera
    readings = []
    def split(image):
        readings.append(image)
        if image[0, 0, 0] == 255:
            raise ValueError("No complete panel frames found")
        return image, _panels(), {}
    monkeypatch.setattr(panel_detection, "split_problem_page", split)
    st.captured = _snapshot()
    assert all(interface.live_camera_workspace())
    old_token = st.session_state.general_camera_capture_provenance["token"]
    st.captured = _snapshot(255)
    assert interface.live_camera_workspace() == ([], [])
    assert "general_camera_capture_provenance" not in st.session_state
    assert st.session_state.general_camera_snapshot_token != old_token
    assert "No complete panel frames" in st.warnings[-1]
    interface.live_camera_workspace()
    assert len(readings) == 2  # Unchanged invalid bytes are not repeatedly parsed.
    assert "No complete panel frames" in st.warnings[-1]


def test_deleting_snapshot_clears_only_snapshot_derived_capture(camera, monkeypatch):
    st, engine = camera
    monkeypatch.setattr(panel_detection, "split_problem_page", lambda image: (image, _panels(50), {}))
    st.captured = _snapshot()
    assert all(interface.live_camera_workspace())
    st.captured = None
    assert interface.live_camera_workspace() == ([], [])
    # An intentional live selection remains available after the video stops.
    engine.panels = _panels(90)
    st.click = True
    with pytest.raises(AppRerun):
        interface.live_camera_workspace()
    engine.reset()
    result = interface.live_camera_workspace()
    assert np.all(result[0][0] == 90)
    assert st.session_state.general_camera_capture_provenance["source"] == "live"


def test_old_snapshot_cannot_overwrite_new_live_selection_on_full_rerun(camera, monkeypatch):
    st, engine = camera
    calls = []
    def split(image):
        calls.append(True)
        return image, _panels(50), {}
    monkeypatch.setattr(panel_detection, "split_problem_page", split)
    st.captured = _snapshot()
    interface.live_camera_workspace()
    engine.panels = _panels(90)
    st.click = True
    with pytest.raises(AppRerun):
        interface.live_camera_workspace()
    result = interface.live_camera_workspace()
    assert np.all(result[0][0] == 90) and len(calls) == 1
    st.captured = None
    assert np.all(interface.live_camera_workspace()[0][0] == 90)
    # A genuinely new snapshot request supersedes even the saved live page.
    st.captured = BytesIO(b"not an image")
    assert interface.live_camera_workspace() == ([], [])


@pytest.mark.parametrize("contents", [b"not an image", b"", b"\x89PNG\r\n\x1a\ntruncated"])
def test_invalid_image_decoding_is_visible_and_never_returns_previous_groups(camera, contents):
    st, engine = camera
    engine.panels = _panels()
    st.click = True
    with pytest.raises(AppRerun):
        interface.live_camera_workspace()
    st.captured = BytesIO(contents)
    assert interface.live_camera_workspace() == ([], [])
    assert st.warnings and "Could not read the camera snapshot" in st.warnings[-1]


def test_snapshot_fallback_operates_without_optional_webrtc_packages(camera, monkeypatch):
    st, _ = camera
    monkeypatch.setitem(sys.modules, "streamlit_webrtc", None)
    monkeypatch.setattr(panel_detection, "split_problem_page", lambda image: (image, _panels(75), {}))
    st.captured = _snapshot()
    result = interface.live_camera_workspace()
    assert np.all(result[0][0] == 75) and not st.buttons
    assert st.session_state.general_camera_capture_provenance["source"] == "snapshot"


def test_real_complete_snapshot_page_is_split_then_removed(camera):
    st, _ = camera
    st.captured = BytesIO((Path(__file__).resolve().parents[1]/"data/fixtures/dotted_page.png").read_bytes())
    positive, negative = interface.live_camera_workspace()
    assert len(positive) == len(negative) == 6 and not st.warnings
    assert all(image.dtype == np.uint8 and image.shape[2] == 3 for image in positive+negative)
    st.captured = None
    assert interface.live_camera_workspace() == ([], [])


def test_transient_snapshot_read_error_clears_old_input_but_can_recover_same_file(camera, monkeypatch):
    st, _ = camera
    monkeypatch.setattr(panel_detection, "split_problem_page", lambda image: (image, _panels(75), {}))
    snapshot = _snapshot()
    st.captured = snapshot
    interface.live_camera_workspace()
    class Unreadable:
        def getvalue(self):
            raise OSError("Snapshot stream unavailable")
    st.captured = Unreadable()
    assert interface.live_camera_workspace() == ([], [])
    assert "Snapshot stream unavailable" in st.warnings[-1]
    st.captured = snapshot
    assert np.all(interface.live_camera_workspace()[0][0] == 75)

from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

from src.interface import image_tasks as interface
from src.reasoning.general import GeneralVisualSolver

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT/"data/fixtures/arrow_paths"


class Upload(BytesIO):
    def __init__(self, name, data):
        super().__init__(data)
        self.name = name


def test_teach_sort_review_export_and_input_invalidation(monkeypatch):
    files = {f"image_task_{letter.lower()}": [Upload(f"{letter}{i}.png", (IMAGES/f"{letter}{i:02d}.png").read_bytes())
                                              for i in range(1, 5)] for letter in "AB"}
    files["image_task_batch"] = [Upload("new-a.png", (IMAGES/"A05.png").read_bytes()),
                                 Upload("new-b.png", (IMAGES/"B05.png").read_bytes()),
                                 Upload("broken.png", b"not pixels")]
    monkeypatch.setattr(DeltaGenerator, "file_uploader", lambda self, *args, key=None, **kwargs: files.get(key, []))
    monkeypatch.setattr(interface.st, "file_uploader", lambda *args, key=None, **kwargs: files.get(key, []))
    monkeypatch.setattr(interface, "model_directory", lambda *args: None)
    fits, original_fit = [], GeneralVisualSolver.fit
    def count_fit(self, *args, **kwargs):
        fits.append(True)
        return original_fit(self, *args, **kwargs)
    monkeypatch.setattr(GeneralVisualSolver, "fit", count_fit)
    app = AppTest.from_file(ROOT/"app.py", default_timeout=45).run()
    app.button(key="nav_image_tasks").click().run()
    app.button(key="image_task_fit_button").click().run()
    app.button(key="image_task_sort").click().run()
    assert not app.exception and not app.error
    saved = app.session_state.image_task_results
    assert [row["automatic_label"] for row in saved["rows"]] == [True, False, None]
    assert len(fits) == 1
    # Review a confident automatic decision, not just an unresolved input.
    selector = next(box for box in app.selectbox if box.label == "Review image")
    selector.set_value(0).run()
    review = next(radio for radio in app.radio if radio.label == "Label for this image")
    review.set_value("Group 2 · Other").run()
    app.button(key="image_task_review_save").click().run()
    assert not app.exception and app.session_state.image_task_results["reviews"] == {"Q0001": False}
    assert len(fits) == 1
    with ZipFile(BytesIO(app.session_state.image_task_results["export"])) as archive:
        report = json.loads(archive.read("results.json"))
        assert report["results"][0]["automatic_label"] is True
        assert report["results"][0]["final_label"] is False
    # A changed batch removes the former results without refitting the references.
    files["image_task_batch"] = [Upload("different.png", (IMAGES/"A06.png").read_bytes())]
    app.run()
    assert len(app.dataframe) == 0 and len(fits) == 1
    app.button(key="image_task_sort").click().run()
    assert len(app.session_state.image_task_results["rows"]) == 1 and len(fits) == 1
    # Changes to the reference task hide results and require an explicit new fit.
    app.text_input(key="image_task_a_name").set_value("Changed group").run()
    assert len(app.dataframe) == 0
    assert not any(button.key == "image_task_sort" for button in app.button)
    assert not app.exception


def test_saved_task_can_be_reopened_and_malformed_input_is_visible(monkeypatch):
    from src.image_tasks import ImageTask
    import numpy as np
    from PIL import Image

    groups = [tuple(np.array(Image.open(IMAGES/f"{letter}{i:02d}.png").convert("RGB")) for i in range(1, 4)) for letter in "AB"]
    task = ImageTask("Saved check", ("Accepted", "Rejected"), "drawings", *groups)
    files = {"image_task_saved": Upload("task.zip", task.export())}
    monkeypatch.setattr(interface.st, "file_uploader", lambda *args, key=None, **kwargs: files.get(key, []))
    app = AppTest.from_file(ROOT/"app.py", default_timeout=45).run()
    app.button(key="nav_image_tasks").click().run()
    app.radio(key="image_task_source").set_value("Open saved setup").run()
    app.button(key="image_task_fit_button").click().run()
    assert not app.exception and not app.error
    assert app.session_state.image_task_fitted[0] == task.fingerprint
    files["image_task_saved"] = Upload("bad.zip", b"broken archive")
    app.run()
    assert app.error and not app.exception
    assert not any(button.key == "image_task_sort" for button in app.button)

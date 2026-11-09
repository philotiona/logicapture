from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
import pytest
from streamlit.testing.v1 import AppTest

from src import image_tasks as tasks
from src.interface import image_tasks as interface
from src.vision import pretrained


def example_sheet(*, missing=False, swapped_colors=False):
    page = Image.new("RGB", (900, 1200), "white")
    draw = ImageDraw.Draw(page)
    centers = []
    for y in (90, 320, 550):
        centers.append([(x, y) for x in (50, 245, 475, 670)])
    centers.append([(150, 845), (565, 845)])
    colors = [(166, 18, 27), (16, 95, 176)]
    if swapped_colors:
        colors.reverse()
    for row, entries in enumerate(centers):
        for col, (x, y) in enumerate(entries):
            if missing and (row, col) == (2, 1):
                continue
            frame = colors[col >= 2] if row < 3 else ((105, 105, 105), colors[1])[col]
            color = (28 + 25 * row, 48 + 15 * col, 128)
            draw.rectangle((x, y, x+179, y+179), fill=color, outline=frame, width=4)
            draw.ellipse((x+24, y+43, x+93, y+122), outline="white", width=3)
    for x in range(55, 840, 20):
        draw.line((x, 785, x+10, 785), fill="orange", width=4)
    draw.text((55, 1080), "Caption with answer and labels", fill="black")
    output = BytesIO()
    page.save(output, format="PNG")
    return output.getvalue()


def test_example_sheet_uses_geometry_and_exports_only_panel_interiors():
    files = tasks.read_example_sheet(example_sheet())
    swapped = tasks.read_example_sheet(example_sheet(swapped_colors=True))
    assert files == swapped
    assert [[name for name, _ in group] for group in files] == [
        [f"a-{i:02d}.png" for i in range(1, 7)], [f"b-{i:02d}.png" for i in range(1, 7)],
        ["test-01.png", "test-02.png"],
    ]
    groups = [tasks.read_images(group)[0] for group in files]
    assert [tuple(item.image[0, 0]) for item in groups[0]] == [
        (28 + 25 * row, 48 + 15 * col, 128) for row in range(3) for col in range(2)
    ]
    assert [tuple(item.image[0, 0]) for item in groups[1]] == [
        (28 + 25 * row, 48 + 15 * col, 128) for row in range(3) for col in range(2, 4)
    ]
    for col, item in enumerate(groups[2]):
        pixels = np.unique(item.image.reshape(-1, 3), axis=0)
        assert {tuple(color) for color in pixels} == {(103, 48 + 15 * col, 128), (255, 255, 255)}


def test_example_sheet_rejects_an_incomplete_grid():
    with pytest.raises(ValueError, match="complete example sheet"):
        tasks.read_example_sheet(example_sheet(missing=True))


class RelationBackend:
    model_id = "test-local-relation"

    def __init__(self, *, failed_check=False):
        self.observed = []
        self.checked = []
        self.failed_check = failed_check

    def complete(self, prompt, images, *, max_new_tokens):
        if not images:
            assert "Return only JSON" in prompt and len(self.observed) == 12
            return '{"condition":"The example has the selected feature."}'
        assert len(images) == 1 and max(images[0].size) < 180
        red, green, _ = images[0].getpixel((0, 0))
        if "Describe what is directly visible" in prompt:
            self.observed.append((red, green))
            return "A visible feature."
        assert "The example has the selected feature." in prompt
        self.checked.append((red, green))
        if self.failed_check and (red, green) == (28, 48):
            return '{"status":"unclear","evidence":"Reference detail is obscured."}'
        if (red, green) == (103, 48) and self.failed_check is None:
            return '{"status":"unclear","evidence":"Query detail is obscured."}'
        present = green < 70 if red < 103 else green == 48
        return json.dumps({"status": "present" if present else "absent", "evidence": "Visible detail."})


@pytest.mark.parametrize("failed_check, expected", [(True, [None, None]), (None, [None, False])])
def test_relationship_fit_abstains_on_failed_references_or_an_unclear_query(failed_check, expected):
    a, b, queries = [tasks.read_images(group)[0] for group in tasks.read_example_sheet(example_sheet())]
    task = tasks.ImageTask("Sheet", ("A", "B"), "photos", tuple(item.image for item in a),
                           tuple(item.image for item in b), photo_relation=True)
    loaded = tasks.load_task(task.export())
    assert loaded.photo_relation and loaded.fingerprint == task.fingerprint
    backend = RelationBackend(failed_check=failed_check)
    fitted = tasks.fit_task(loaded, photo_backend=backend)
    assert len(backend.checked) == 12 and all(red < 103 for red, _ in backend.checked)
    rows = tasks.sort_batch(fitted, queries)
    assert [row["automatic_label"] for row in rows] == expected
    assert len(backend.checked) == (12 if failed_check else 14)


def test_example_sheet_import_sort_and_download(monkeypatch):
    page = BytesIO(example_sheet())
    page.name = "sheet.png"
    upload = {"image_task_sheet": page}
    backend = RelationBackend()
    monkeypatch.setattr(interface.st, "file_uploader", lambda *args, key=None, **kwargs: upload.get(key))
    monkeypatch.setattr(interface, "model_directory", lambda name: True)
    monkeypatch.setattr(pretrained, "QwenLocalBackend", lambda: backend)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    app.button(key="nav_image_tasks").click().run()
    app.radio(key="image_task_source").set_value("Import an example sheet").run()
    assert "6 left references, 6 right references, and 2 test images" in app.success[0].value
    assert app.radio(key="image_task_photo_method_sheet").value == "Action or relationship"
    app.button(key="image_task_fit_button").click().run()
    assert len(backend.checked) == 12
    app.button(key="image_task_sort").click().run()
    assert not app.error and not app.exception
    saved = app.session_state.image_task_results
    assert [row["automatic_label"] for row in saved["rows"]] == [True, False]
    with ZipFile(BytesIO(saved["export"])) as result:
        report = json.loads(result.read("results.json"))
        for row, (name, data) in zip(report["results"], tasks.read_example_sheet(page.getvalue())[2]):
            assert row["filename"] == name
            assert result.read(row["export_path"]) == data
            assert row["evidence"]["visible_evidence"] == "Visible detail."
    upload["image_task_sheet"] = BytesIO(example_sheet(missing=True))
    app.run()
    assert app.error and not app.exception and not app.dataframe

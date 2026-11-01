from io import BytesIO
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from src.interface import rgr as interface
from src.rgr import photos, report

ROOT = Path(__file__).resolve().parents[1]


def _image():
    image = np.full((60, 100, 3), 245, np.uint8)
    image[15:45, 25:75] = 40
    image[15:23, 25:40] = 120
    return image


def _file(image, name="photo.png"):
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return SimpleNamespace(name=name, getvalue=lambda: stream.getvalue())


def test_photo_decode_respects_transparency_and_exif_orientation():
    rgba = np.zeros((16, 24, 4), np.uint8)
    rgba[5:10, 7:12] = (255, 0, 0, 255)
    decoded = photos.decode_photo(_file(rgba).getvalue())
    assert tuple(decoded[0, 0]) == (255, 255, 255)
    assert tuple(decoded[7, 9]) == (255, 0, 0)
    stream = BytesIO()
    image = Image.fromarray(_image())
    exif = Image.Exif()
    exif[274] = 6
    image.save(stream, format="PNG", exif=exif)
    oriented = photos.decode_photo(stream.getvalue())
    assert oriented.shape == (100, 60, 3)
    np.testing.assert_array_equal(oriented, np.rot90(_image(), k=3))


def test_photo_decode_rejects_invalid_large_and_animated_inputs(monkeypatch):
    for invalid in (b"not an image", b"", "not bytes"):
        with pytest.raises(ValueError):
            photos.decode_photo(invalid)
    frames = [Image.new("RGB", (20, 20), color) for color in ("white", "black")]
    stream = BytesIO()
    frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    with pytest.raises(ValueError, match="still image"):
        photos.decode_photo(stream.getvalue())
    monkeypatch.setattr(photos, "MAX_PIXELS", 500)
    with pytest.raises(ValueError, match="megapixels"):
        photos.decode_photo(_file(_image()).getvalue())
    monkeypatch.setattr(photos, "MAX_BYTES", 4)
    with pytest.raises(ValueError, match="15 MB"):
        photos.decode_photo(b"12345")


def test_manual_crop_is_reproducible_and_preserves_original_pixel_values():
    image = _image()
    cropped, box = photos.crop_photo(image, (20, 80), (10, 90))
    assert box == (20, 6, 80, 54)
    np.testing.assert_array_equal(cropped, image[6:54, 20:80])
    cropped[:] = 0
    assert image[7, 21, 0] == 245
    for limits in ((80, 20), (0, 0), (-1, 90), (0, 101), (0, float("nan")), None):
        with pytest.raises(ValueError):
            photos.crop_photo(image, horizontal=limits)
    with pytest.raises(ValueError, match="8 pixels"):
        photos.crop_photo(image, (20, 21))


class Model:
    def __call__(self, batch, training=False):
        return {"shape": np.tile([.1, .1, .7, .1], (len(batch), 1))}


def test_single_photo_needs_no_dataset_and_changed_pixels_invalidate_measurements(monkeypatch):
    monkeypatch.setattr(interface, "_checkpoint", lambda path: Model())
    state = {"file": _file(_image())}
    real_uploader = interface.st.file_uploader
    def uploader(*args, **kwargs):
        return state["file"] if kwargs.get("key") == "rgr_single_upload" else real_uploader(*args, **kwargs)
    monkeypatch.setattr(interface.st, "file_uploader", uploader)
    exports = []
    original_export = report.build_export
    def capture_export(payload, images=None):
        exports.append(payload)
        return original_export(payload, images)
    monkeypatch.setattr(report, "build_export", capture_export)
    app = AppTest.from_file(ROOT/"app.py", default_timeout=35).run()
    app.button(key="nav_rgr").click().run()
    app.radio(key="rgr_source").set_value("Single photograph").run()
    assert not app.exception and not app.error
    assert not any(b.key == "rgr_train" for b in app.button)
    assert exports[-1]["metadata"]["dataset_kind"] == "single_photo"
    assert "correct" not in exports[-1]["single_image"][0]
    assert "initial" not in exports[-1] and "final" not in exports[-1]
    app.button(key="rgr_measure_photo").click().run()
    assert not app.exception and exports[-1]["sweeps"]
    assert all("correct" not in row for rows in exports[-1]["sweeps"].values() for row in rows)
    original_id = interface._image_id(_image())
    app.slider(key="rgr_crop_x_"+original_id[:16]).set_value((20, 80)).run()
    assert not exports[-1]["sweeps"]
    assert exports[-1]["metadata"]["crop_box_pixels"] == (20, 0, 80, 60)
    app.selectbox(key="rgr_truth_"+original_id[:16]).set_value("quadrilateral").run()
    app.button(key="rgr_measure_photo").click().run()
    assert exports[-1]["single_image"][0]["correct"] is True
    changed = _image()
    changed[0, 0] = 5
    state["file"] = _file(changed)  # Same filename, different canonical pixels.
    app.run()
    assert not app.exception and not exports[-1]["sweeps"]
    assert exports[-1]["metadata"]["known_shape"] is None
    assert exports[-1]["metadata"]["crop_box_pixels"] == (0, 0, 100, 60)
    state["file"] = None
    app.run()
    assert not app.exception and not any(b.key == "rgr_measure_photo" for b in app.button)
    assert any("Upload a photograph" in info.value for info in app.info)


def test_single_photo_export_does_not_claim_unknown_truth_or_five_image_evaluation():
    from zipfile import ZipFile
    payload = {"metadata": {"dataset_kind": "single_photo"}, "single_image": [
        {"image_id": "one", "image_name": "one.png", "prediction": "circle", "confidence": .8,
         "accepted": True, "elapsed_ms": 3.}]}
    with ZipFile(BytesIO(report.build_export(payload))) as archive:
        markdown = archive.read("REPORT.md").decode("utf-8")
        coverage = archive.read("requirement_coverage.csv").decode("utf-8-sig")
        assert "## Single photograph" in markdown
        assert "one.png | not supplied | circle" in markdown
        assert "one.png | not supplied | circle | 0.8 | not assessed" in markdown
        assert "A single-photo experiment" in coverage

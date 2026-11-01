from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from streamlit.testing.v1 import AppTest

from src.interface import rgr as interface
from src.rgr.dataset import Sample, demo_dataset, split_samples

ROOT = Path(__file__).resolve().parents[1]


class ShapeModel:
    def __call__(self, batch, training=False):
        return {"shape": np.tile([.15, .2, .6, .05], (len(batch), 1))}


def test_rgr_sliders_experiments_and_report_use_current_pixels(monkeypatch):
    monkeypatch.setattr(interface, "_checkpoint", lambda path: ShapeModel())
    app = AppTest.from_file(ROOT/"app.py", default_timeout=35).run()
    app.button(key="nav_rgr").click().run()
    assert not app.exception and not app.error
    assert app.slider(key="rgr_threshold").value == .6
    assert app.slider(key="rgr_brightness").value == 0
    assert app.slider(key="rgr_contrast").value == 1.
    app.button(key="rgr_run_sweeps").click().run()
    assert not app.exception and not app.error
    saved = app.session_state.rgr_sweeps
    assert {row["value"] for row in saved["data"]["size"]} == {128, 224, 320, 416, 640}
    assert len(saved["data"]["augmentation"]) == 3
    assert any("measured results" in expander.label for expander in app.expander)
    app.button(key="rgr_compare").click().run()
    assert not app.exception and not app.error
    comparison = app.session_state.rgr_comparison
    assert len(comparison["initial"]["before"]) == len(comparison["final"]["before"]) == 5
    assert {row["image_id"] for row in comparison["initial"]["before"]}.isdisjoint(
        row["image_id"] for row in comparison["final"]["before"])
    assert [row["image_id"] for row in comparison["initial"]["before"]] == [
        row["image_id"] for row in comparison["initial"]["after"]]
    app.slider(key="rgr_brightness").set_value(100).run()
    assert not app.exception
    assert any("blank or uniform" in warning.value for warning in app.warning)
    assert not any("measured results" in expander.label for expander in app.expander)
    assert any("changed" in caption.value for caption in app.caption)


def test_cohorts_exclude_duplicate_pixels_and_use_separate_sources():
    samples = split_samples(demo_dataset(30))["test"]
    copied = Sample("copy.png", samples[0].image.copy(), samples[0].label)
    first, final = interface._cohorts([samples[0], copied]+samples[1:])
    assert len(first) == len(final) == 5
    identifiers = [interface._image_id(s.image) for s in first+final]
    assert len(set(identifiers)) == 10


def test_report_exports_only_current_measurements():
    from src.rgr.report import build_export
    image = np.zeros((8, 8, 3), np.uint8)
    data = build_export({"metadata": {"dataset_kind": "synthetic"}, "sweeps": {}}, {"original.png": image})
    with ZipFile(BytesIO(data)) as archive:
        payload = json.loads(archive.read("analysis.json"))
        assert payload["metadata"]["dataset_kind"] == "synthetic"
        assert "REPORT.md" in archive.namelist()
        assert not payload.get("sweeps")


def test_ui_comparison_uses_the_verified_training_seed(monkeypatch):
    from src.rgr.dataset import pixel_digest
    monkeypatch.setattr(interface, "_checkpoint", lambda path: ShapeModel())
    def recorded_run(directory, samples):
        return {"seed": 2027, "selected_epoch": 1, "adapted_model_sha256": "a"*64}, split_samples(samples, seed=2027)
    monkeypatch.setattr(interface, "_available_training", recorded_run)
    app = AppTest.from_file(ROOT/"app.py", default_timeout=35).run()
    app.button(key="nav_rgr").click().run()
    app.button(key="rgr_compare").click().run()
    assert not app.exception and not app.error
    comparison = app.session_state.rgr_comparison
    split = split_samples(demo_dataset(30), seed=2027)
    forbidden = {pixel_digest(sample.image) for key in ("train", "validation") for sample in split[key]}
    seen = {row["image_id"] for key in ("initial", "final") for row in comparison[key]["before"]}
    assert len(seen) == 10 and seen.isdisjoint(forbidden)
    assert seen <= {pixel_digest(sample.image) for sample in split["test"]}

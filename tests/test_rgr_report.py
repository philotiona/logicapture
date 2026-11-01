from io import BytesIO
import csv
import json
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

from src.rgr.report import build_export


def _read_csv(archive, path):
    return list(csv.DictReader(archive.read(path).decode("utf-8-sig").splitlines()))


def _rows(prefix, correct):
    return [{"image_id": f"{prefix}{i}", "true_label": "triangle",
             "predicted_label": "triangle" if i < correct else "circle", "confidence": .8}
            for i in range(5)]


def test_export_preserves_readings_and_computes_real_same_image_results():
    payload = {"metadata": {"source": "Жеке суреттер", "dataset_kind": "personal_photos"},
               "initial": {"before": _rows("initial", 2), "after": list(reversed(_rows("initial", 4)))},
               "final": {"before": _rows("final", 3), "after": _rows("final", 5), "slider_count": 3}}
    image = np.full((8, 9, 3), 180, np.uint8)
    exported = build_export(payload, {f"before_after_{i}": image for i in range(3)})
    assert exported == build_export(payload, {f"before_after_{i}": image for i in range(3)})
    with ZipFile(BytesIO(exported)) as archive:
        assert json.loads(archive.read("analysis.json")) == payload
        summary = json.loads(archive.read("derived_summaries.json"))
        assert summary["initial"]["before"]["accuracy"] == .4
        assert summary["initial"]["after"]["accuracy"] == .8
        assert summary["final"]["after"]["correct"] == 5
        assert summary["final"]["after"]["errors"] == 0
        pairs = _read_csv(archive, "tables/initial_paired.csv")
        assert len(pairs) == 5 and pairs[0]["image_id"] == "initial0"
        assert json.loads(pairs[0]["after"])["image_id"] == "initial0"
        coverage = {row["task"]: row for row in _read_csv(archive, "requirement_coverage.csv")}
        assert coverage["9"]["status"] == "complete"
        assert coverage["10"]["status"] == "complete"
        report = archive.read("REPORT.md").decode("utf-8")
        assert "Жеке суреттер" in report and "40.00%" in report and "80.00%" in report
        assert "OpenCV geometric classifier" in report and "Compact attribute CNN" in report
        with Image.open(BytesIO(archive.read("illustrations/before_after_0.png"))) as saved:
            np.testing.assert_array_equal(saved, image)


def test_missing_truth_training_and_personal_photos_are_not_claimed_complete():
    payload = {"metadata": {"dataset_kind": "synthetic"},
               "initial": {"before": [{"image_id": "a", "confidence": .99}],
                           "after": [{"image_id": "a", "confidence": .99}]}}
    with ZipFile(BytesIO(build_export(payload))) as archive:
        summary = json.loads(archive.read("derived_summaries.json"))["initial"]["before"]
        assert summary["accuracy"] is None and summary["unassessed"] == 1
        coverage = _read_csv(archive, "requirement_coverage.csv")
        assert coverage[0]["status"] == "complete"
        assert all(row["status"] == "incomplete" for row in coverage[1:])
        assert "Synthetic examples" in archive.read("REPORT.md").decode("utf-8") or "synthetic demonstration" in archive.read("REPORT.md").decode("utf-8")


def test_sweep_coverage_requires_all_actual_settings_and_size_timing():
    reading = {"prediction": "circle", "confidence": .6, "accepted": True}
    payload = {"sweeps": {
        "threshold": [{"threshold": v, **reading} for v in (.1, .3, .5, .7, .9)],
        "brightness": [{"brightness": v, **reading} for v in (-100, -50, 0, 50, 100)],
        "contrast": [{"contrast": v, **reading} for v in (.5, 1., 2.)],
        "size": [{"size": v, **reading} for v in (128, 224, 320, 416, 640)],
        "augmentation": [{"augmentation": v, **reading}
                         for v in ("original", "rotate", "blur")],
    }}
    with ZipFile(BytesIO(build_export(payload))) as archive:
        coverage = {row["task"]: row["status"] for row in _read_csv(archive, "requirement_coverage.csv")}
        assert all(coverage[key] == "complete" for key in ("3", "4", "5", "7"))
        assert coverage["6"] == "incomplete"
        assert len(_read_csv(archive, "tables/sweeps_threshold.csv")) == 5


def test_cohort_overlap_and_missing_pairing_remain_incomplete():
    payload = {"metadata": {"dataset_kind": "personal_photos"},
               "initial": {"before": _rows("same", 3), "after": _rows("same", 5)},
               "final": {"before": _rows("same", 3), "after": _rows("same", 5), "slider_count": 3}}
    with ZipFile(BytesIO(build_export(payload))) as archive:
        coverage = {row["task"]: row for row in _read_csv(archive, "requirement_coverage.csv")}
        assert coverage["10"]["status"] == "incomplete"
        assert json.loads(coverage["10"]["measured"])["distinct_from_initial"] is False


def test_csv_formula_text_is_escaped_but_json_and_numeric_values_are_preserved():
    payload = {"sweeps": {"brightness": [{"brightness": -100, "label": " =HYPERLINK(\"bad\")",
                                           "prediction": "@SUM(1,2)", "truth": "Сурет"}]}}
    with ZipFile(BytesIO(build_export(payload))) as archive:
        row = _read_csv(archive, "tables/sweeps_brightness.csv")[0]
        assert row["brightness"] == "-100"
        assert row["label"].startswith("'") and row["prediction"].startswith("'")
        assert row["truth"] == "Сурет"
        assert json.loads(archive.read("analysis.json")) == payload


def test_numpy_measurements_are_serializable_and_nonfinite_readings_are_explicitly_missing():
    payload = {"metadata": {"score": np.float32(.5), "unavailable": np.nan, "indices": np.array([1, 2])}}
    with ZipFile(BytesIO(build_export(payload))) as archive:
        metadata = json.loads(archive.read("analysis.json"))["metadata"]
        assert metadata == {"score": .5, "unavailable": None, "indices": [1, 2]}


def test_real_application_contract_reports_saved_adaptation_and_rejections_separately():
    initial = _rows("initial", 5)
    initial[0].update(accepted=False, correct=False)
    initial[1].update(accepted=True, correct=False, predicted_label="circle")
    payload = {"metadata": {"dataset_kind": "personal_photos", "sliders": ["confidence", "brightness", "contrast"]},
               "initial": {"before": initial, "after": _rows("initial", 5)},
               "final": {"after": _rows("new", 5)},
               "training": {"split_counts": {"train": 20, "validation": 8, "test": 12},
                            "history": [{"epoch": 1}], "selected_epoch": 1, "base_model_sha256": "abc"},
               "sweeps": {"augmentation": [{"value": v, "prediction": "circle", "confidence": .8, "accepted": True}
                                            for v in ("original", "rotate20", "hflip")]}}
    image = np.zeros((3, 3, 3), np.uint8)
    pictures = {f"before_after/{i}_{state}.png": image for i in range(3) for state in ("before", "after")}
    with ZipFile(BytesIO(build_export(payload, pictures))) as archive:
        summary = json.loads(archive.read("derived_summaries.json"))["initial"]["before"]
        assert summary["rejected"] == 1 and summary["errors"] == 1 and summary["incorrect_or_rejected"] == 2
        assert summary["accuracy"] == .6
        coverage = {row["task"]: row for row in _read_csv(archive, "requirement_coverage.csv")}
        assert all(coverage[key]["status"] == "complete" for key in ("7", "8", "9", "10"))
        assert json.loads(coverage["9"]["measured"])["comparison_illustrations"] == 3


def test_sweep_setting_lists_without_inference_readings_remain_incomplete():
    payload = {"sweeps": {"threshold": [{"value": value} for value in (.1, .3, .5, .7, .9)],
                          "brightness": [{"value": value} for value in (-100, -50, 0, 50, 100)],
                          "contrast": [{"value": value} for value in (.5, 1, 2)],
                          "size": [{"value": value, "elapsed_ms": 1} for value in (128, 224, 320, 416, 640)]}}
    with ZipFile(BytesIO(build_export(payload))) as archive:
        coverage = {row["task"]: row for row in _read_csv(archive, "requirement_coverage.csv")}
        assert all(coverage[key]["status"] == "incomplete" for key in ("3", "4", "5", "6"))


@pytest.mark.parametrize("invalid", [
    {"confidence": -.1}, {"confidence": 1.1}, {"confidence": float("nan")},
    {"confidence": "0.8"}, {"confidence": True}, {"accepted": 1},
    {"accepted": None}, {"prediction": None, "accepted": True},
])
def test_invalid_sweep_outputs_do_not_satisfy_assignment_coverage(invalid):
    rows = [{"value": value, "prediction": "triangle", "accepted": True, "confidence": .8}
            for value in (.1, .3, .5, .7, .9)]
    rows[2].update(invalid)
    with ZipFile(BytesIO(build_export({"sweeps": {"threshold": rows}}))) as archive:
        coverage = {row["task"]: row for row in _read_csv(archive, "requirement_coverage.csv")}
        assert coverage["3"]["status"] == "incomplete"


def test_explicit_no_class_rejections_are_valid_sweep_readings():
    rows = [{"value": value, "prediction": None, "accepted": False, "confidence": 0.}
            for value in (.1, .3, .5, .7, .9)]
    with ZipFile(BytesIO(build_export({"sweeps": {"threshold": rows}}))) as archive:
        coverage = {row["task"]: row for row in _read_csv(archive, "requirement_coverage.csv")}
        assert coverage["3"]["status"] == "complete"
        assert "no class (rejected)" in archive.read("REPORT.md").decode("utf-8")


@pytest.mark.parametrize("timing", [-1, float("inf"), None, "1.5", True])
def test_size_sweep_requires_finite_nonnegative_numeric_timing(timing):
    rows = [{"value": value, "prediction": "triangle", "accepted": True, "confidence": .8, "elapsed_ms": 1.}
            for value in (128, 224, 320, 416, 640)]
    rows[2]["elapsed_ms"] = timing
    with ZipFile(BytesIO(build_export({"sweeps": {"size": rows}}))) as archive:
        coverage = {row["task"]: row for row in _read_csv(archive, "requirement_coverage.csv")}
        assert coverage["6"]["status"] == "incomplete"


def test_report_derives_parameter_conclusions_from_recorded_values_only():
    def reading(value, prediction="triangle", confidence=.8, accepted=True, **extra):
        return {"value": value, "prediction": prediction, "confidence": confidence, "accepted": accepted, **extra}
    payload = {"sweeps": {
        "threshold": [reading(.1), reading(.9, accepted=False)],
        "brightness": [reading(-100, None, 0., False), reading(0), reading(100, "circle", .6)],
        "contrast": [reading(.5, confidence=.6), reading(1, confidence=.8), reading(2, confidence=.7, accepted=False)],
        "size": [reading(128, confidence=.8, elapsed_ms=3), reading(224, confidence=.9, elapsed_ms=2),
                 reading(640, confidence=.7, elapsed_ms=4)],
        "augmentation": [reading("original", confidence=.8, truth="triangle"),
                         reading("rotate20", "circle", .6, truth="triangle")],
    }, "training": {"before": {"accuracy": .4, "mean_confidence": .8},
                     "after": {"accuracy": .6, "mean_confidence": .7}}}
    with ZipFile(BytesIO(build_export(payload))) as archive:
        report = archive.read("REPORT.md").decode("utf-8")
    assert "accepted threshold values were 0.1; rejected threshold values were 0.9" in report
    assert "-100 → 0: no class (rejected) → triangle" in report
    assert "0 → 100: triangle → circle" in report
    assert "Reported confidence ranged from 0.6000 to 0.8000" in report
    assert "Confidence did not improve monotonically" in report
    assert "Fastest recorded single run: input size 224, 2.000 ms" in report
    assert "rotate20: prediction circle; confidence 0.6000; confidence change versus original -0.2000; recorded result incorrect" in report
    assert "raw test accuracy increased: 40.00% before → 60.00% after (+20.00 percentage points)" in report
    assert "mean confidence changed from 0.8000 to 0.7000 (-0.1000)" in report
    assert "selected image only" in report and "not calibrated accuracy" in report


@pytest.mark.parametrize("confidences, expected", [
    ([.6, .7, .8], "was nondecreasing"),
    ([.8, .8, .8], "stayed unchanged"),
    ([.8, .7, .6], "decreased or stayed equal"),
])
def test_size_conclusion_describes_actual_confidence_direction(confidences, expected):
    rows = [{"value": size, "prediction": "triangle", "confidence": confidence, "accepted": True}
            for size, confidence in zip((128, 224, 640), confidences)]
    with ZipFile(BytesIO(build_export({"sweeps": {"size": rows}}))) as archive:
        report = archive.read("REPORT.md").decode("utf-8")
    assert expected in report
    assert "No finite nonnegative runtimes were supplied" in report
    assert "Fastest recorded single run" not in report


def test_missing_readings_do_not_produce_fabricated_conclusions_or_augmentation_deltas():
    payload = {"sweeps": {"threshold": [{"value": .5}],
                          "augmentation": [{"value": "rotate20", "prediction": "circle", "confidence": .7, "accepted": True}]},
               "training": {"epochs": 1}}
    with ZipFile(BytesIO(build_export(payload))) as archive:
        report = archive.read("REPORT.md").decode("utf-8")
    assert "No complete inference readings were supplied" in report
    assert "No complete original-image baseline was supplied" in report
    assert "no accuracy-change conclusion is drawn" in report
    assert "accepted threshold values were" not in report
    assert "confidence change versus original" not in report
    assert "raw test accuracy increased" not in report


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:\\drive", "ok/../../outside", "bad\x00name"])
def test_unsafe_illustration_names_are_rejected(name):
    with pytest.raises(ValueError, match="Unsafe illustration"):
        build_export({}, {name: np.zeros((3, 3, 3), np.uint8)})


def test_invalid_records_image_pixels_and_duplicate_normalized_names_have_distinct_errors():
    with pytest.raises(TypeError, match="payload must be a dictionary"):
        build_export([])
    with pytest.raises(ValueError, match="initial.before must be a list"):
        build_export({"initial": {"before": "invented measurements"}})
    with pytest.raises(ValueError, match="uint8"):
        build_export({}, {"bad": np.zeros((3, 3, 3), np.float32)})
    image = np.zeros((3, 3, 3), np.uint8)
    with pytest.raises(ValueError, match="Duplicate normalized illustration"):
        build_export({}, {"comparison.jpg": image, "comparison.png": image})

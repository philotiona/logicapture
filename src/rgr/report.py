"""In-memory, evidence-based RGR report export; no fabricated evaluation scores."""
from __future__ import annotations

import csv
from io import BytesIO, StringIO
import json
import math
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image


MODEL_COMPARISON = [
    {"name": "Compact attribute CNN", "task": "Shape/fill classification of segmented objects",
     "advantages": "Trainable on personal examples; reports class confidence; existing project-pretrained checkpoint",
     "disadvantages": "Depends on segmentation; domain shift; confidence is not calibrated accuracy",
     "resources": "TensorFlow/Keras; local checkpoint; 113,558 parameters; modest object crops",
     "project_fit": "Chosen: reuses the project's attribute model and supports before/after fine-tuning"},
    {"name": "MobileNetV2", "task": "General image classification / transferable image features",
     "advantages": "Compact general-purpose architecture; official Keras implementation; transferable features",
     "disadvantages": "ImageNet labels do not directly classify this project's geometric shapes; requires a new task head and labeled adaptation",
     "resources": "TensorFlow/Keras; additional weights if ImageNet initialization is used",
     "project_fit": "Considered, not trained or benchmarked here; less direct reuse than the existing narrow shape CNN"},
    {"name": "OpenCV geometric classifier", "task": "Contour-based shape/fill classification",
     "advantages": "Local, deterministic, lightweight; no training data or neural weights",
     "disadvantages": "Fixed geometry rules; broken/merged contours and irregular drawings can fail",
     "resources": "CPU; OpenCV; no learned checkpoint",
     "project_fit": "Existing geometric alternative and useful baseline; cannot be fine-tuned with gradients"},
]

_IDENTIFIERS = ("image_digest", "source_hash", "image_hash", "image_id", "image", "filename", "name", "id")
_TRUTH = ("true_label", "expected_label", "ground_truth", "label_true", "truth", "true")
_PREDICTION = ("predicted_label", "prediction", "predicted", "label_pred", "pred")
_CONFIDENCE = ("confidence", "mean_confidence", "avg_confidence")


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def _first(row, names):
    return next((row[name] for name in names if name in row and row[name] is not None), None)


def _identity(row):
    value = _first(row, _IDENTIFIERS)
    return None if value is None else str(value)


def _correct(row):
    explicit = _first(row, ("correct", "is_correct"))
    if isinstance(explicit, bool):
        return explicit
    if row.get("accepted") is False:
        return False
    truth, prediction = _first(row, _TRUTH), _first(row, _PREDICTION)
    if truth is not None and prediction is not None:
        return truth == prediction
    if row.get("expected_count") is not None:
        count = _first(row, ("detected_count", "object_count", "count"))
        if count is not None:
            return count == row["expected_count"]
    return None


def _summary(rows):
    checks = [_correct(row) for row in rows]
    assessed = [value for value in checks if value is not None]
    confidences = [_first(row, _CONFIDENCE) for row in rows]
    confidences = [float(value) for value in confidences
                   if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)]
    return {"images": len(rows), "assessed": len(assessed), "correct": sum(assessed),
            "errors": sum(check is False and row.get("accepted") is not False for row, check in zip(rows, checks)),
            "rejected": sum(row.get("accepted") is False for row in rows),
            "incorrect_or_rejected": len(assessed)-sum(assessed), "unassessed": len(rows)-len(assessed),
            "accuracy": sum(assessed)/len(assessed) if assessed else None,
            "mean_confidence": sum(confidences)/len(confidences) if confidences else None,
            "confidence_records": len(confidences)}


def _rows(value, label):
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{label} must be a list of record dictionaries")
    return value


def _cohort(payload, name):
    value = payload.get(name, {})
    if name == "initial" and not value:
        value = payload.get("comparison", {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} comparison must be a dictionary")
    return {**value, "before": _rows(value.get("before"), f"{name}.before"),
            "after": _rows(value.get("after"), f"{name}.after")}


def _paired(cohort):
    before, after = cohort["before"], cohort["after"]
    left, right = [_identity(row) for row in before], [_identity(row) for row in after]
    valid = (bool(left) and all(left) and all(right) and len(set(left)) == len(left)
             and len(set(right)) == len(right) and set(left) == set(right))
    if not valid:
        return [], False
    after_by_id = {_identity(row): row for row in after}
    return [{"image_id": _identity(row), "before": row, "after": after_by_id[_identity(row)]}
            for row in before], True


def _cell(value):
    text = _json(value) if isinstance(value, (dict, list)) else "" if value is None else str(value)
    # CSV quoting alone does not stop formulas when opened in spreadsheet apps.
    if isinstance(value, str) and (re.match(r"^\s*[=+@-]", text) or text.startswith(("\t", "\r", "\n"))):
        text = "'"+text
    return text


def _csv(rows):
    stream = StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow([_cell(column) for column in columns])
    for row in rows:
        writer.writerow([_cell(row.get(column)) for column in columns])
    return stream.getvalue().encode("utf-8-sig")


def _slug(value):
    return re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("._") or "records"


def _tables(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _tables(item, (*prefix, _slug(key)))
    elif isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
        yield "_".join(prefix) or "records", value


def _illustration_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Illustration names must be nonempty strings")
    path = PurePosixPath(name.replace("\\", "/"))
    if (path.is_absolute() or any(part in ("..", ".") for part in path.parts)
            or ":" in name or any(ord(char) < 32 for char in name)):
        raise ValueError(f"Unsafe illustration name: {name!r}")
    if not path.parts:
        raise ValueError("Illustration names must identify a file")
    return "illustrations/"+str(path.with_suffix(".png"))


def _png(value, name):
    array = np.asarray(value)
    if (array.dtype != np.uint8 or array.ndim not in (2, 3)
            or (array.ndim == 3 and array.shape[2] not in (1, 3, 4))
            or not array.size or min(array.shape[:2]) < 1):
        raise ValueError(f"Illustration {name!r} requires nonempty uint8 grayscale, RGB or RGBA pixels")
    if array.shape[0]*array.shape[1] > 80_000_000:
        raise ValueError(f"Illustration {name!r} exceeds the 80-megapixel limit")
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    image = Image.fromarray(array)
    if image.mode == "RGBA":
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image)
    stream = BytesIO()
    image.convert("RGB").save(stream, format="PNG")
    return stream.getvalue()


def _values(rows, names):
    values = set()
    for row in rows:
        value = _first(row, names)
        if isinstance(value, list):
            values.add(tuple(value))
        elif isinstance(value, (int, float, str)):
            values.add(value)
    return values


def _finite_number(value, minimum=None, maximum=None):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (minimum is None or value >= minimum)
            and (maximum is None or value <= maximum))


def _complete_reading(row):
    confidence = _first(row, _CONFIDENCE)
    predicted = next((row[key] for key in _PREDICTION if key in row), None)
    return (_finite_number(confidence, 0, 1) and isinstance(row.get("accepted"), bool)
            and any(key in row for key in _PREDICTION)
            and (predicted is not None or row["accepted"] is False))


def _coverage(payload, cohorts, illustration_names):
    metadata = payload.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    photos = metadata.get("dataset_kind") == "personal_photos"
    initial, final = cohorts["initial"], cohorts["final"]
    initial_pairs, same_initial = _paired(initial)
    final_pairs, same_final = _paired(final)
    initial_ids = {_identity(row) for row in initial["before"] if _identity(row)}
    final_ids = {_identity(row) for row in final["after"] if _identity(row)}
    independent = bool(initial_ids and final_ids and not (initial_ids & final_ids))
    sweeps = payload.get("sweeps", {})
    if not isinstance(sweeps, dict):
        raise ValueError("sweeps must be a dictionary")
    sweeps = {key: _rows(value, f"sweeps.{key}") for key, value in sweeps.items()}
    training = payload.get("training") or {}
    if not isinstance(training, dict):
        raise ValueError("training must be a dictionary")
    splits = training.get("splits", training.get("split_counts", training.get("split", {})))
    splits = splits if isinstance(splits, dict) else {}
    validation = splits.get("val", splits.get("validation"))
    split_ok = all((isinstance(value, int) and not isinstance(value, bool) and value > 0)
                   or (isinstance(value, list) and bool(value))
                   for value in (splits.get("train"), validation, splits.get("test")))
    saved = bool(training.get("saved_model") or training.get("model_path") or training.get("checkpoint") or training.get("model_saved"))
    # The application only supplies a persisted training.json after verifying
    # its adjacent model.keras and dataset identity. Its record has no path key.
    saved = saved or bool(training.get("selected_epoch") and training.get("history") and training.get("base_model_sha256"))
    rows = []

    def add(number, requirement, measured, complete, note=""):
        rows.append({"task": number, "requirement": requirement,
                     "status": "complete" if complete else "incomplete", "measured": measured, "note": note})

    add(1, "Compare at least two models and justify the choice", len(MODEL_COMPARISON), True,
        "Compact attribute CNN chosen; MobileNetV2 considered, not benchmarked; OpenCV geometric baseline. Qualitative comparison, no invented benchmark scores.")
    before_summary = _summary(initial["before"])
    add(2, "Pretrained model on at least five new personal photographs with truth, prediction, confidence and correctness",
        before_summary, photos and len(initial["before"]) >= 5 and before_summary["unassessed"] == 0
        and before_summary["confidence_records"] == len(initial["before"]),
        ("A single-photo experiment does not provide the required five independent test photographs."
         if metadata.get("dataset_kind") == "single_photo" else
         "Synthetic examples demonstrate the workflow but do not satisfy the personal-photo requirement.")
        if not photos else "Photo provenance is user-declared; novelty is not independently verified.")
    requirements = [
        (3, "threshold", {0.1, 0.3, 0.5, 0.7, 0.9}, ("threshold", "value")),
        (4, "brightness", {-100, -50, 0, 50, 100}, ("brightness", "value", "delta")),
        (5, "contrast", {0.5, 1, 2}, ("contrast", "value", "factor")),
        (6, "size", {128, 224, 320, 416, 640}, ("size", "input_size", "side", "value")),
    ]
    for number, name, expected, keys in requirements:
        records = sweeps.get(name, [])
        observed = _values(records, keys)
        readings = all(_complete_reading(row) for row in records)
        timing = name != "size" or all(_finite_number(
            _first(row, ("time_ms", "elapsed_ms", "latency_ms", "seconds", "elapsed_seconds")), 0)
            for row in records)
        add(number, f"{name.title()} sweep at {', '.join(map(str, sorted(expected)))}"+(" with timing" if name == "size" else ""),
            {"values": sorted(observed, key=str), "records": len(records), "complete_readings": readings,
             "valid_timing": timing if name == "size" else None}, expected.issubset(observed) and readings and timing,
            "Classification confidence threshold; this is not a detector's bounding-box threshold." if name == "threshold" else "")
    augmentation = sweeps.get("augmentation", [])
    variants = _values(augmentation, ("augmentation", "variant", "transform", "name", "value"))
    original = any(str(value).casefold() in ("original", "none", "identity") for value in variants)
    add(7, "Original plus at least two augmentations with predictions and confidence", sorted(variants, key=str),
        original and len(variants) >= 3 and all(_complete_reading(row) for row in augmentation),
        "An explicit rejection/no-class result is a measured outcome, not an invented prediction.")
    add(8, "Personal dataset, train/validation/test split, fine-tuning and saved model",
        {"split_counts": splits, "model_saved": saved, "training_recorded": bool(training)},
        photos and split_ok and saved and bool(training),
        "Evaluation photos must be excluded from fitting and augmentation of training data.")
    paired_visuals = set()
    for name in illustration_names:
        if "before_after" not in name.casefold() and "before-after" not in name.casefold():
            continue
        stem = str(PurePosixPath(name).with_suffix(""))
        match = re.match(r"^(.*)_(before|after)$", stem, re.IGNORECASE)
        if match:
            other = match[1]+("_after.png" if match[2].casefold() == "before" else "_before.png")
            if other in illustration_names:
                paired_visuals.add(match[1])
        else:
            paired_visuals.add(stem)
    after_summary = _summary(initial["after"])
    add(9, "Same five test photographs before/after and at least three visual Before → After comparisons",
        {"same_image_pairs": len(initial_pairs), "comparison_illustrations": len(paired_visuals)},
        photos and same_initial and len(initial_pairs) >= 5 and len(paired_visuals) >= 3
        and after_summary["unassessed"] == 0 and after_summary["confidence_records"] == len(initial["after"]),
        "Pairing requires stable unique image IDs or image digests in both model results. Static comparisons are included; interactive views remain in the application.")
    final_summary = _summary(final["after"])
    slider_names = metadata.get("sliders", [])
    declared_sliders = len(slider_names) if isinstance(slider_names, (list, tuple)) else 0
    sliders = final.get("slider_count", metadata.get("final_slider_count", declared_sliders))
    add(10, "Five new final test photographs, correct/errors/mean confidence/Accuracy, and at least three sliders",
        {**final_summary, "distinct_from_initial": independent, "declared_slider_count": sliders,
         "same_image_before_after_pairs": len(final_pairs)},
        photos and len(final["after"]) >= 5 and independent
        and final_summary["unassessed"] == 0 and final_summary["confidence_records"] == len(final["after"])
        and isinstance(sliders, (int, float)) and sliders >= 3,
        "Slider count is application evidence declared in the payload; ZIP exports contain static images. Image IDs must be disjoint from the initial cohort.")
    return rows


def _markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _result_text(row):
    if row.get("accepted") is False:
        return "rejected"
    correct = _correct(row)
    return "not assessed" if correct is None else "correct" if correct else "incorrect"


def _prediction_text(row):
    prediction = _first(row, _PREDICTION)
    return "no class (rejected)" if prediction is None and row.get("accepted") is False else prediction


def _setting_value(row, name):
    return _first(row, ("value", str(name), "variant", "transform", "name", "size", "input_size", "side"))


def _runtime_ms(row):
    milliseconds = _first(row, ("elapsed_ms", "time_ms", "latency_ms"))
    if _finite_number(milliseconds, 0):
        return float(milliseconds)
    seconds = _first(row, ("seconds", "elapsed_seconds"))
    return float(seconds)*1000 if _finite_number(seconds, 0) else None


def _sweep_observations(name, records):
    """Describe actual readings without treating one image as an accuracy test."""
    measured = [row for row in records if _complete_reading(row) and _setting_value(row, name) is not None]
    if not measured:
        return ["No complete inference readings were supplied for this sweep; no data-specific conclusion is drawn."]
    display = lambda value: _markdown_cell("not recorded" if value is None else value)
    settings = lambda rows: ", ".join(display(_setting_value(row, name)) for row in rows) or "none"
    confidence = [float(_first(row, _CONFIDENCE)) for row in measured]
    scope = "These readings concern the selected image only; they do not establish dataset accuracy."
    if name == "threshold":
        accepted = [row for row in measured if row["accepted"]]
        rejected = [row for row in measured if not row["accepted"]]
        return [f"On the selected image, accepted threshold values were {settings(accepted)}; rejected threshold values were {settings(rejected)}. "
                "This measures confidence filtering, rather than recall or accuracy over a dataset. "+scope]
    if name in ("brightness", "contrast"):
        ordered = sorted(measured, key=lambda row: (not _finite_number(_setting_value(row, name)),
                         float(_setting_value(row, name)) if _finite_number(_setting_value(row, name)) else str(_setting_value(row, name))))
        transitions = [(left, right) for left, right in zip(ordered, ordered[1:])
                       if _first(left, _PREDICTION) != _first(right, _PREDICTION)]
        if transitions:
            changes = "; ".join(f"{display(_setting_value(left, name))} → {display(_setting_value(right, name))}: "
                                f"{display(_prediction_text(left))} → {display(_prediction_text(right))}"
                                for left, right in transitions)
            classes = "Class/no-class transitions were "+changes+"."
        elif all(_first(row, _PREDICTION) is None for row in ordered):
            classes = "No tested setting produced a class prediction."
        else:
            classes = "The raw predicted class stayed unchanged across the recorded settings."
        return [classes+f" Rejected values were {settings([row for row in measured if not row['accepted']])}. "
                f"Reported confidence ranged from {min(confidence):.4f} to {max(confidence):.4f}. "
                "Confidence is uncalibrated and is not evidence of correctness. "+scope]
    if name == "size":
        ordered = sorted((row for row in measured if _finite_number(_setting_value(row, name))),
                         key=lambda row: float(_setting_value(row, name)))
        sizes = [_setting_value(row, name) for row in ordered]
        values = [float(_first(row, _CONFIDENCE)) for row in ordered]
        if len(set(sizes)) < 2:
            trend = "Fewer than two distinct measured sizes were supplied; a confidence trend cannot be established."
        else:
            differences = [right-left for left, right in zip(values, values[1:])]
            if all(abs(value) <= 1e-12 for value in differences):
                trend = "Confidence stayed unchanged as the tested input sizes increased; larger sizes did not increase it on this image."
            elif all(value >= -1e-12 for value in differences):
                trend = "Confidence was nondecreasing as the tested input sizes increased, with increases at some sizes."
            elif all(value <= 1e-12 for value in differences):
                trend = "Confidence decreased or stayed equal as the tested input sizes increased."
            else:
                trend = "Confidence did not improve monotonically as the tested input sizes increased."
        timed = [(float(_runtime_ms(row)), row) for row in measured if _runtime_ms(row) is not None]
        if timed:
            fastest, row = min(timed, key=lambda item: item[0])
            speed = f"Fastest recorded single run: input size {display(_setting_value(row, name))}, {fastest:.3f} ms. "
            speed += "One timing run does not establish stable throughput or a speed guarantee; hardware, caching and warm-up affect it."
        else:
            speed = "No finite nonnegative runtimes were supplied; a speed comparison cannot be made."
        return [trend+" "+speed+" Confidence is not calibrated accuracy. "+scope]
    if name == "augmentation":
        original = next((row for row in measured if str(_setting_value(row, name)).casefold()
                         in ("original", "none", "identity")), None)
        observations = []
        if original is None:
            observations.append("No complete original-image baseline was supplied; confidence deltas cannot be calculated.")
        baseline = float(_first(original, _CONFIDENCE)) if original is not None else None
        for row in measured:
            value = float(_first(row, _CONFIDENCE))
            text = f"{display(_setting_value(row, name))}: prediction {display(_prediction_text(row))}; confidence {value:.4f}"
            if baseline is not None:
                text += f"; confidence change versus original {value-baseline:+.4f}"
            if _first(row, _TRUTH) is not None or isinstance(_first(row, ("correct", "is_correct")), bool):
                text += "; recorded result "+_result_text(row)
            elif row["accepted"] is False:
                text += "; rejected"
            observations.append(text+".")
        observations.append("Confidence changes do not establish accuracy changes. "+scope)
        return observations
    return [scope]


def _training_observations(training):
    before, after = training.get("before", {}), training.get("after", {})
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    accuracy_before, accuracy_after = before.get("accuracy"), after.get("accuracy")
    if _finite_number(accuracy_before, 0, 1) and _finite_number(accuracy_after, 0, 1):
        change = float(accuracy_after)-float(accuracy_before)
        direction = "increased" if change > 1e-12 else "decreased" if change < -1e-12 else "stayed unchanged"
        result = [f"Training-record raw test accuracy {direction}: {accuracy_before:.2%} before → {accuracy_after:.2%} after "
                  f"({change*100:+.2f} percentage points). These are the recorded test-split values, not an estimate for all future images."]
    else:
        result = ["Paired finite training accuracy values were not supplied; no accuracy-change conclusion is drawn."]
    confidence_before, confidence_after = before.get("mean_confidence"), after.get("mean_confidence")
    if _finite_number(confidence_before, 0, 1) and _finite_number(confidence_after, 0, 1):
        result.append(f"Recorded mean confidence changed from {confidence_before:.4f} to {confidence_after:.4f} "
                      f"({confidence_after-confidence_before:+.4f}). A confidence change is separate from an accuracy change; the scores are uncalibrated.")
    return result


def _report(payload, cohorts, summaries, coverage, illustration_names):
    metadata = payload.get("metadata", {})
    lines = ["# RGR: reproducible computer vision report", "",
             "This export records the measurements actually supplied by the application. Missing work is marked incomplete; no benchmark scores are inferred from the model choice.", "",
             "## Project reuse and model choice", "",
             "The implementation reuses the Bongard project's object segmentation, project-pretrained shape/fill attribute CNN, geometric OpenCV classifier, and local inference. The compact CNN is selected because it already fits the project's object labels and supports adapting the shape head. Frozen-feature shape-head adaptation is transfer/adaptation within this project; it is not training a new general-purpose vision model. MobileNetV2 is considered as a pretrained neural alternative and is not trained or benchmarked in this report. ImageNet classes do not directly provide the project's geometric labels. The geometric pipeline provides a lightweight baseline. Their strengths are qualitative unless measured records are present.", "",
             "| Model | Task | Advantages | Disadvantages | Resources | Fit and choice |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in MODEL_COMPARISON:
        lines.append("| "+" | ".join(_markdown_cell(row[key]) for key in ("name", "task", "advantages", "disadvantages", "resources", "project_fit"))+" |")
    lines.extend(["", "Architecture references: [official Keras MobileNetV2 API](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2), [official TensorFlow transfer-learning guide](https://www.tensorflow.org/tutorials/images/transfer_learning). These describe the considered architecture and adaptation approach; they are not experimental scores.", "", "## Dataset and settings", "", "```json", _json(metadata), "```", "",
                  "A synthetic demonstration is not evidence that the personal-photo assignment is complete. Photo provenance and ground-truth annotations must be supplied by the author. Train, validation and test images must remain separate.", "",
                  "## Measured results", "", "Accuracy below uses only rows with explicit correctness or an available truth/prediction pair. Rejected cases count against accuracy and are listed separately from accepted classification errors. Unassessed rows are reported separately. Confidence is the model's reported score, not calibrated probability of correctness.", "",
                  "| Cohort | Model state | Images | Assessed | Correct | Errors | Rejected | Unassessed | Accuracy | Mean confidence |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for name, values in summaries.items():
        for state, summary in values.items():
            accuracy = "not measured" if summary["accuracy"] is None else f"{summary['accuracy']:.2%}"
            confidence = "not measured" if summary["mean_confidence"] is None else f"{summary['mean_confidence']:.4f}"
            lines.append(f"| {name} | {state} | {summary['images']} | {summary['assessed']} | {summary['correct']} | {summary['errors']} | {summary['rejected']} | {summary['unassessed']} | {accuracy} | {confidence} |")
    single = _rows(payload.get("single_image"), "single_image")
    if single:
        lines.extend(["", "## Single photograph", "",
                      "This photograph is a separate parameter experiment. It was not added to training or to the initial/final evaluation cohorts.", "",
                      "| Image | Truth | Prediction | Confidence | Result | Time (ms) |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for row in single:
            values = [row.get("image_name", _identity(row)), _first(row, _TRUTH), _prediction_text(row),
                      _first(row, _CONFIDENCE), _result_text(row), row.get("elapsed_ms")]
            lines.append("| "+" | ".join(_markdown_cell("not supplied" if value is None else value) for value in values)+" |")
    for name, cohort in cohorts.items():
        paired, valid = _paired(cohort)
        lines.extend(["", f"### {name.title()} cohort before/after", "",
                      f"Stable same-image pairing: {'verified from supplied identifiers' if valid else 'not established'}. Pairs: {len(paired)}."])
        if paired:
            lines.extend(["", "| Image | Truth | Before prediction | Before confidence | Before result | After prediction | After confidence | After result |",
                          "| --- | --- | --- | --- | --- | --- | --- | --- |"])
            for pair in paired:
                before, after = pair["before"], pair["after"]
                values = [before.get("image_name", pair["image_id"]), _first(before, _TRUTH), _prediction_text(before),
                          _first(before, _CONFIDENCE), _result_text(before), _prediction_text(after),
                          _first(after, _CONFIDENCE), _result_text(after)]
                lines.append("| "+" | ".join(_markdown_cell("not measured" if value is None else value) for value in values)+" |")
        else:
            for state in ("before", "after"):
                if not cohort[state]:
                    continue
                lines.extend(["", f"{state.title()} per-image readings:", "",
                              "| Image | Truth | Prediction | Confidence | Result |",
                              "| --- | --- | --- | --- | --- |"])
                for row in cohort[state]:
                    values = [_identity(row), _first(row, _TRUTH), _prediction_text(row),
                              _first(row, _CONFIDENCE), _result_text(row)]
                    lines.append("| "+" | ".join(_markdown_cell("not measured" if value is None else value) for value in values)+" |")
        for state in ("before", "after"):
            supplied = cohort.get("metrics_"+state)
            if supplied is not None:
                lines.extend(["", f"Additional application metrics ({state}; preserved as supplied):", "", "```json", _json(supplied), "```"])
    sweeps = payload.get("sweeps", {})
    if sweeps:
        lines.extend(["", "## Parameter experiments", "",
                      "Each row is an actual pipeline reading. Sweep values change one parameter at a time. Rejections retain their raw class prediction when available."])
        for name, records in sweeps.items():
            if not records:
                continue
            lines.extend(["", f"### {_markdown_cell(name).title()}", "",
                          "| Value | Prediction | Confidence | Accepted | Time (ms) |",
                          "| --- | --- | --- | --- | --- |"])
            for row in records:
                values = [_first(row, ("value", str(name), "variant", "transform", "name")),
                          _prediction_text(row), _first(row, _CONFIDENCE), row.get("accepted"),
                          _first(row, ("elapsed_ms", "time_ms", "latency_ms"))]
                lines.append("| "+" | ".join(_markdown_cell("not measured" if value is None else value) for value in values)+" |")
            for observation in _sweep_observations(name, records):
                lines.extend(["", observation])
    training = payload.get("training") or {}
    if training:
        fields = ("source", "adaptation", "metrics_semantics", "epochs", "selected_epoch", "checkpoint_selection",
                  "trainable_parameters", "split_counts", "augmentation", "before", "after", "model_path", "limitations")
        lines.extend(["", "## Recorded adaptation", "", "```json",
                      _json({key: training[key] for key in fields if key in training}), "```", "",
                      "Epoch-by-epoch readings and source split manifests are preserved in analysis.json and the training CSV tables. Validation selects the checkpoint; test metrics do not select an epoch."])
        for observation in _training_observations(training):
            lines.extend(["", observation])
    lines.extend(["", "## Assignment coverage", "", "| Task | Requirement | Status | Evidence / limitations |",
                  "| --- | --- | --- | --- |"])
    for row in coverage:
        lines.append("| "+" | ".join(_markdown_cell(value) for value in (row["task"], row["requirement"], row["status"],
                     json.dumps(row["measured"], ensure_ascii=False)+" "+row["note"]))+" |")
    lines.extend(["", "## Interpretation and limits", "",
                  "Threshold experiments filter class-confidence outputs; they are not object-detector localization experiments. Size timing describes the actual pipeline settings recorded in the tables. The CNN may internally resize crops to its fixed training resolution, so page/input size does not imply a changed neural architecture.", "",
                  "Image transformations can alter pixel measurements and classification. Fine-tuning can improve or worsen unseen results; use the recorded before/after values and independent final cohort to judge it. Small cohorts and uncalibrated confidence limit generalization claims. Training-record accuracy uses unthresholded argmax predictions; the interactive comparison tables apply the selected confidence threshold.", "",
                  "The ZIP includes analysis.json with the original readings, derived_summaries.json, requirement_coverage.csv, and CSV tables for supplied record lists. Empty or absent sections are not silently replaced with example measurements."])
    if illustration_names:
        lines.extend(["", "## Supplied illustrations", ""])
        for name in illustration_names:
            lines.extend([f"### {_markdown_cell(PurePosixPath(name).stem)}", "", f"![Supplied comparison]({quote(name, safe='/')})", ""])
    return "\n".join(lines)+"\n"


def build_export(payload: dict, illustrations: dict[str, np.ndarray] | None = None) -> bytes:
    """Return a reproducible ZIP containing machine readings, CSVs and REPORT.md.

    ``initial`` and ``final`` comparison dictionaries may contain ``before`` /
    ``after`` record lists and supplied metrics. Missing cohorts remain visibly
    incomplete. Images must have safe relative names and uint8 pixel arrays.
    """
    if not isinstance(payload, dict):
        raise TypeError("Report payload must be a dictionary")
    if illustrations is not None and not isinstance(illustrations, dict):
        raise TypeError("Illustrations must be a name-to-image dictionary")
    payload = _plain(payload)
    cohorts = {name: _cohort(payload, name) for name in ("initial", "final")}
    summaries = {name: {state: _summary(cohort[state]) for state in ("before", "after")}
                 for name, cohort in cohorts.items()}
    files = {"analysis.json": _json(payload).encode("utf-8"),
             "derived_summaries.json": _json(summaries).encode("utf-8")}
    illustration_names = []
    for name, pixels in (illustrations or {}).items():
        target = _illustration_name(name)
        if target in files:
            raise ValueError(f"Duplicate normalized illustration name: {target}")
        files[target] = _png(pixels, name)
        illustration_names.append(target)
    coverage = _coverage(payload, cohorts, illustration_names)
    files["requirement_coverage.csv"] = _csv(coverage)
    for name, rows in _tables(payload):
        target = "tables/"+name+".csv"
        if target in files:
            raise ValueError(f"Duplicate normalized table name: {target}")
        files[target] = _csv(rows)
    files["tables/model_comparison.csv"] = _csv(MODEL_COMPARISON)
    for name, cohort in cohorts.items():
        paired, _ = _paired(cohort)
        if paired:
            files[f"tables/{name}_paired.csv"] = _csv(paired)
    files["REPORT.md"] = _report(payload, cohorts, summaries, coverage, illustration_names).encode("utf-8")
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for name, contents in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, contents)
    return stream.getvalue()

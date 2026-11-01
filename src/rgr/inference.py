"""Shared, pixel-preserving inference for the RGR single-shape classifier.

The existing CNN has a fixed 48×48 grayscale input and four shape classes.
The requested processing size controls the intermediate image, not the model's
input architecture. No segmentation or automatic contrast normalization can
undo a user's brightness and contrast settings here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

from src.vision.attribute_cnn import SHAPE_CLASSES, normalize_region
from src.vision.pretrained import rgb_image

ROOT = Path(__file__).resolve().parents[2]
CLASSES = tuple(SHAPE_CLASSES)
MODEL_PATH = ROOT / "data" / "models" / "attributes.keras"
SIZES = (128, 224, 320, 416, 640)


@dataclass(frozen=True)
class Settings:
    threshold: float = .6
    brightness: float = 0
    contrast: float = 1
    size: int = 224
    rotation: float = 0
    flip: bool = False

    def __post_init__(self):
        for name in ("threshold", "brightness", "contrast", "rotation"):
            value = getattr(self, name)
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if not -100 <= self.brightness <= 100:
            raise ValueError("Brightness must be between -100 and 100 percent.")
        if not .5 <= self.contrast <= 2:
            raise ValueError("Contrast must be between 0.5 and 2.")
        if isinstance(self.size, bool) or self.size not in SIZES:
            raise ValueError(f"Processing size must be one of {SIZES}.")
        if not isinstance(self.flip, (bool, np.bool_)):
            raise ValueError("Horizontal flip must be Boolean.")


def _rgb(image):
    picture = rgb_image(image)
    if min(picture.size) < 3:
        raise ValueError("Use an image at least 3×3 pixels.")
    if picture.width * picture.height > 80_000_000:
        raise ValueError("Image exceeds the 80-megapixel limit.")
    return picture


def transform_image(image, settings=Settings()):
    """Return the visible RGB intermediate image used by the CNN pipeline.

    Contrast is linear around mid-gray; brightness is additive in percent of
    full scale. Values are clipped only at the final uint8 conversion. Padding
    and rotation fill begin white and receive the same adjustments as pixels.
    """
    picture = _rgb(image)
    side = int(settings.size)
    scale = side / max(picture.size)
    resized = picture.resize((max(1, round(picture.width * scale)),
                              max(1, round(picture.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(resized, ((side-resized.width)//2, (side-resized.height)//2))
    if settings.flip:
        canvas = ImageOps.mirror(canvas)
    if settings.rotation:
        canvas = canvas.rotate(float(settings.rotation), resample=Image.Resampling.BILINEAR,
                               expand=False, fillcolor="white")
    values = (np.asarray(canvas, dtype=np.float32)-127.5)*float(settings.contrast)+127.5
    values += float(settings.brightness)*255/100
    return np.rint(np.clip(values, 0, 255)).astype(np.uint8)


def prepare_model_input(image, settings=Settings()):
    """Common training/inference preparation: an unbatched 48×48×1 tensor."""
    return normalize_region(transform_image(image, settings))


def load_model(path=MODEL_PATH):
    """Load an existing local checkpoint; TensorFlow is imported only here."""
    from src.vision.attribute_cnn import load_attribute_model
    return load_attribute_model(str(Path(path).resolve()))


def _unusable(picture):
    gray = np.asarray(picture.convert("L") if isinstance(picture, Image.Image)
                      else Image.fromarray(picture).convert("L"), dtype=np.float32)
    # Uniform white/black/gray inputs are not members of the shape vocabulary.
    # Reject near-uniform and one-pixel noise without an adaptive threshold.
    return float(np.ptp(gray)) < 4 or np.count_nonzero(np.abs(gray-np.median(gray)) >= 4) < 4


def predict_image(image, model, settings=Settings()):
    """Measure a genuine shape-head prediction, retaining rejected argmaxes.

    A low-confidence result preserves its raw predicted class. A uniform or
    saturated image has no class prediction and never invokes the CNN.
    Malformed model outputs fail explicitly instead of inventing confidence.
    """
    started = perf_counter()
    original = _rgb(image)
    processed = transform_image(original, settings)
    result = {"prediction": None, "accepted": False, "confidence": 0., "probabilities": {},
              "model_input_size": [48, 48], "processed_size": int(settings.size)}
    if _unusable(original) or _unusable(processed):
        result["reason"] = "blank_or_uniform_image"
    else:
        batch = normalize_region(processed)[None, ...]
        output = model(batch, training=False)
        if not isinstance(output, dict) or "shape" not in output:
            raise ValueError("The model must expose its named shape probability head.")
        probabilities = np.asarray(output["shape"], dtype=np.float64)
        if probabilities.shape != (1, len(CLASSES)):
            raise ValueError("The model shape head must return one probability per shape class.")
        probabilities = probabilities[0]
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1., atol=1e-4)):
            raise ValueError("The model shape head returned invalid softmax probabilities.")
        winner = int(np.argmax(probabilities))
        confidence = float(probabilities[winner])
        result.update(prediction=CLASSES[winner], confidence=confidence,
                      probabilities=dict(zip(CLASSES, map(float, probabilities))),
                      accepted=confidence >= settings.threshold)
        if not result["accepted"]:
            result["reason"] = "below_confidence_threshold"
    result["elapsed_ms"] = (perf_counter()-started)*1000
    return result


def evaluate_images(images, labels, model, settings=Settings()):
    """Return per-image decisions; rejected cases count as incorrect."""
    images, labels = list(images), list(labels)
    if len(images) != len(labels):
        raise ValueError("Each evaluation image needs exactly one ground-truth label.")
    if any(label not in CLASSES for label in labels):
        raise ValueError(f"Ground-truth labels must belong to {CLASSES}.")
    rows = []
    for index, (image, truth) in enumerate(zip(images, labels), start=1):
        result = predict_image(image, model, settings)
        rows.append({"index": index, "truth": truth, **result,
                     "correct": result["accepted"] and result["prediction"] == truth})
    return rows


def metrics(rows):
    """Accuracy uses every image; accepted errors and rejections are separate."""
    rows = list(rows)
    total = len(rows)
    accepted = sum(bool(row["accepted"]) for row in rows)
    correct = sum(bool(row["correct"]) for row in rows)
    raw_correct = sum(row["prediction"] == row["truth"] for row in rows)
    return {"total": total, "correct": correct, "errors": accepted-correct,
            "rejected": total-accepted, "accuracy": correct/total if total else 0.,
            "mean_confidence": float(np.mean([row["confidence"] for row in rows])) if rows else 0.,
            "coverage": accepted/total if total else 0.,
            "raw_accuracy": raw_correct/total if total else 0.,
            "accepted_accuracy": correct/accepted if accepted else 0.}


def parameter_sweeps(image, model, settings=Settings()):
    """Vary one setting at a time through the exact same prediction pipeline."""
    cases = {
        "threshold": [(value, replace(settings, threshold=value)) for value in (.1, .3, .5, .7, .9)],
        "brightness": [(value, replace(settings, brightness=value)) for value in (-100, -50, 0, 50, 100)],
        "contrast": [(value, replace(settings, contrast=value)) for value in (.5, 1., 2.)],
        "size": [(value, replace(settings, size=value)) for value in SIZES],
        "augmentation": [("original", replace(settings, rotation=0, flip=False)),
                         ("rotate20", replace(settings, rotation=20, flip=False)),
                         ("hflip", replace(settings, rotation=0, flip=True))],
    }
    return {name: [{"value": value, **predict_image(image, model, variant)} for value, variant in variants]
            for name, variants in cases.items()}

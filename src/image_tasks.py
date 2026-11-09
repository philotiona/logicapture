"""Reusable example-defined image tasks and lossless, reviewable batch exports.

Task ZIPs contain JSON and PNG references, never executable or pickled models.
Opening a task refits its references with the current local implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO
import csv
import json
from pathlib import PurePosixPath
import re
import stat
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile
from zlib import error as ZlibError

import numpy as np
from PIL import Image

from src.vision.pretrained import image_digest, rgb_image

MAX_FILES = 100
MAX_BYTES = 100 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_SIDE = 1280
MAX_REFERENCES = 12
MAX_REFERENCE_PIXELS = 30_000_000
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
FORMAT = "logicapture-image-task"
VERSION = 1
REVIEW_REASONS = {
    "contradictory_support": "The reference groups contain conflicting evidence",
    "insufficient_support_for_validation": "There are too few references to cross-check this distinction",
    "weak_support_validation": "The references do not separate reliably when cross-checked",
    "small_decision_margin": "This image is close to the boundary between the two groups",
    "unstable_support_fit": "The decision changes when some references are withheld",
    "outside_support_neighborhood": "This image looks unlike the reference examples",
    "unresolved_structural_evidence": "The required drawing features could not be measured",
    "structural_rules_disagree": "The candidate drawing rules disagree on this image",
    "unresolved_measurement_or_program_disagreement": "A required measurement is missing or the candidate rules disagree",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def _safe_member(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ":" in name
            or ".." in path.parts or "\x00" in name):
        raise ValueError("The ZIP contains an unsafe file path.")


def _archive(data):
    if len(data) > MAX_BYTES:
        raise ValueError("ZIP exceeds the 100 MB limit.")
    try:
        with ZipFile(BytesIO(data)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) > MAX_FILES + 1 or sum(item.file_size for item in members) > MAX_BYTES:
                raise ValueError("ZIP exceeds the file count or 100 MB expanded-size limit.")
            names = set()
            for item in members:
                _safe_member(item.orig_filename)
                if item.filename in names or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("ZIP contains duplicate paths or symbolic links.")
                if item.flag_bits & 1:
                    raise ValueError("Encrypted ZIPs are unsupported.")
                names.add(item.filename)
            return [(item.filename, archive.read(item)) for item in members]
    except (BadZipFile, RuntimeError, NotImplementedError, OSError, ZlibError, EOFError) as error:
        raise ValueError("Could not read this ZIP archive.") from error


def decode_image(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Image is empty or exceeds the 100 MB limit.")
    try:
        with Image.open(BytesIO(data)) as original:
            if original.width * original.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 24-megapixel limit.")
            if min(original.size) < 3 or getattr(original, "n_frames", 1) != 1:
                raise ValueError("Use one static image at least 3×3 pixels in size.")
            image = rgb_image(original)
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            if min(image.size) < 3:
                raise ValueError("Image is too narrow to analyze after scaling. Crop it or split it into smaller images.")
            return np.array(image)
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Could not decode this image.") from error


@dataclass
class ImageItem:
    name: str
    data: bytes
    image: np.ndarray | None
    error: str | None = None

    @property
    def digest(self):
        return sha256(self.data).hexdigest()


def read_images(files):
    """Read individual files or ZIPs in memory, retaining originals and failures."""
    files = list(files)
    if len(files) > MAX_FILES or sum(len(data) for _, data in files) > MAX_BYTES:
        raise ValueError("Use at most 100 files totaling at most 100 MB.")
    entries = []
    for name, data in files:
        entries.extend(_archive(data) if name.lower().endswith(".zip") else [(name, data)])
    if len(entries) > MAX_FILES or sum(len(data) for _, data in entries) > MAX_BYTES:
        raise ValueError("Use at most 100 files totaling at most 100 MB after expanding ZIPs.")
    items, skipped, pixels = [], [], 0
    for name, data in entries:
        if PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES:
            skipped.append(name)
            continue
        try:
            image, error = decode_image(data), None
        except (ValueError, TypeError) as problem:
            image, error = None, str(problem)
        if image is not None:
            pixels += image.shape[0]*image.shape[1]
            if pixels > 60_000_000:
                raise ValueError("Batch exceeds 60 million analysis pixels. Use a smaller batch.")
        items.append(ImageItem(name, data, image, error))
    return items, skipped


@dataclass
class ImageTask:
    name: str
    groups: tuple[str, str]
    representation: str
    positive: tuple[np.ndarray, ...]
    negative: tuple[np.ndarray, ...]

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 120
                or len(self.groups) != 2 or any(not isinstance(g, str) or not g.strip() or len(g) > 60 for g in self.groups)
                or self.groups[0].strip().casefold() == self.groups[1].strip().casefold()):
            raise ValueError("Give the task a name and two distinct group names (up to 60 characters each).")
        if self.representation not in ("drawings", "photos"):
            raise ValueError("Choose drawings or photographs.")
        for group in (self.positive, self.negative):
            if not 2 <= len(group) <= MAX_REFERENCES:
                raise ValueError("Use 2–12 reference images per group; 4–6 varied examples are preferable.")
        images = self.positive+self.negative
        if any(not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
               or image.shape[2] != 3 or min(image.shape[:2]) < 3 or max(image.shape[:2]) > MAX_SIDE for image in images):
            raise ValueError("References need RGB analysis pixels, with each side between 3 and 1280 pixels.")
        if sum(image.shape[0]*image.shape[1] for image in images) > MAX_REFERENCE_PIXELS:
            raise ValueError("References exceed 30 million analysis pixels. Use fewer or smaller reference images.")
        digests = [[image_digest(image) for image in group] for group in (self.positive, self.negative)]
        if len(set(digests[0])) != len(digests[0]) or len(set(digests[1])) != len(digests[1]):
            raise ValueError("A group contains repeated reference pixels. Add distinct examples.")
        if set(digests[0]) & set(digests[1]):
            raise ValueError("The same reference image is labeled in both groups. Correct its label.")

    def manifest(self):
        return {"format": FORMAT, "version": VERSION, "name": self.name, "groups": list(self.groups),
                "representation": self.representation,
                "references": [{"file": f"references/{'a' if label else 'b'}{index+1:02d}.png",
                                "label": label, "pixel_sha256": image_digest(image)}
                               for label, group in ((True, self.positive), (False, self.negative))
                               for index, image in enumerate(group)]}

    @property
    def fingerprint(self):
        return sha256(_json(self.manifest())).hexdigest()

    def export(self):
        stream = BytesIO()
        with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
            archive.writestr("task.json", _json(self.manifest()))
            for entry, image in zip(self.manifest()["references"], self.positive + self.negative):
                encoded = BytesIO()
                Image.fromarray(image).save(encoded, format="PNG")
                archive.writestr(entry["file"], encoded.getvalue())
        return stream.getvalue()


def load_task(data):
    try:
        entries = dict(_archive(data))
        if len(entries.get("task.json", b"")) > 64*1024:
            raise ValueError("The saved task manifest is too large.")
        manifest = json.loads(entries["task.json"])
        if (manifest["format"] != FORMAT or type(manifest["version"]) is not int
                or manifest["version"] != VERSION):
            raise ValueError("Unsupported image-task format or version.")
        references = manifest["references"]
        if not 4 <= len(references) <= 2*MAX_REFERENCES:
            raise ValueError("A saved task needs 2–12 references per group.")
        positive, negative, used = [], [], set()
        for entry in references:
            path, label = entry["file"], entry["label"]
            if not isinstance(path, str) or type(label) is not bool or path in used or not path.startswith("references/"):
                raise ValueError("Invalid or repeated task reference.")
            image = decode_image(entries[path])
            if image_digest(image) != entry["pixel_sha256"]:
                raise ValueError("A saved reference does not match its recorded pixels.")
            (positive if label else negative).append(image)
            used.add(path)
        if set(entries) != {"task.json", *used}:
            raise ValueError("Saved task contains unexpected files.")
        return ImageTask(manifest["name"], tuple(manifest["groups"]), manifest["representation"],
                         tuple(positive), tuple(negative))
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid image-task archive or manifest.") from error


def fit_task(task, *, semantic_encoder=None, progress=None):
    from src.reasoning.general import GeneralVisualSolver
    from src.vision.structural_features import StructuralEncoder
    if task.representation == "photos" and semantic_encoder is None:
        from src.vision.pretrained import SiglipEncoder
        semantic_encoder = SiglipEncoder()
    solver = GeneralVisualSolver(semantic_encoder if task.representation == "photos" else None,
                                 StructuralEncoder() if task.representation == "drawings" else None,
                                 structural_first=task.representation == "drawings")
    return solver.fit(task.positive, task.negative, progress=progress)


def sort_batch(fitted, items, *, progress=None):
    """Freeze decisions before human review; do not train on batch images."""
    rows, pending = [], []
    for index, item in enumerate(items):
        pixels = image_digest(item.image) if item.image is not None else None
        reason = item.error or ("Matches a reference image; not an independent prediction."
                                if pixels in fitted.support_digests else None)
        rows.append({"id": f"Q{index+1:04d}", "filename": item.name, "original_sha256": item.digest,
                     "input_sha256": pixels, "automatic_label": None, "suggested_label": None, "score": None,
                     "reason": reason, "error": item.error is not None, "evidence": {}, "nearest_reference_ids": []})
        if reason is None:
            pending.append(index)
    for start in range(0, len(pending), 8):
        indices = pending[start:start+8]
        if progress:
            progress(f"Checking new images {start+1}–{start+len(indices)} of {len(pending)}")
        result = fitted.predict([items[i].image for i in indices], image_ids=[rows[i]["id"] for i in indices])
        for index, prediction in zip(indices, result.predictions):
            rows[index].update(automatic_label=prediction.label, suggested_label=prediction.forced_label, score=prediction.score,
                               reason="; ".join(REVIEW_REASONS.get(reason, reason.replace("_", " ")) for reason in prediction.abstention_reasons)
                                      or "The fitted task accepted this decision.",
                               evidence=prediction.diagnostics, nearest_reference_ids=list(prediction.nearest_support_ids))
    return rows


def _filename(name):
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:100]
    return value or "image"


def export_results(task, fitted, items, rows, reviews=None):
    """Export exact original bytes. Reviews change output labels, never the fit."""
    reviews = reviews or {}
    if fitted.support_digests != frozenset(image_digest(image) for image in task.positive+task.negative):
        raise ValueError("The fitted task does not match these reference images.")
    if len(items) != len(rows) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Results must match the batch images.")
    if set(reviews) - {row["id"] for row in rows} or any(value is not None and type(value) is not bool for value in reviews.values()):
        raise ValueError("Reviews must name batch images and use a group label or None.")
    stream, records = BytesIO(), []
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        for index, (item, row) in enumerate(zip(items, rows)):
            if item.digest != row["original_sha256"]:
                raise ValueError("The batch has changed since these predictions. Sort it again.")
            reviewed = row["id"] in reviews
            label = reviews[row["id"]] if reviewed else row["automatic_label"]
            # Unreadable images remain errors even if a human accidentally sets a label.
            label = None if row["error"] else label
            folder = "errors" if row["error"] else "needs-review" if label is None else f"{'01' if label else '02'}-{_filename(task.groups[0 if label else 1])}"
            destination = f"{folder}/{index+1:04d}-{_filename(PurePosixPath(item.name.replace(chr(92), '/')).name)}"
            archive.writestr(destination, item.data)
            records.append({**row, "reviewed": reviewed, "review_label": reviews.get(row["id"]),
                            "final_label": label, "final_group": task.groups[0 if label else 1] if label is not None else "Needs review",
                            "decision_source": "error" if row["error"] else "human" if reviewed else "automatic",
                            "export_path": destination})
        archive.writestr("results.json", _json({"task": task.manifest(), "fit": fitted.diagnostics,
            "encoder_id": fitted.encoder_id, "task_fingerprint": task.fingerprint,
            "policy": "References only select the fit. Human reviews affect exports only. Scores are not probabilities.",
            "results": records}))
        text = StringIO(newline="")
        fields = ("id", "filename", "automatic_group", "suggested_group", "final_group", "decision_source", "reason", "score", "export_path", "original_sha256")
        writer = csv.DictWriter(text, fieldnames=fields)
        writer.writeheader()
        for record in records:
            label = record["automatic_label"]
            suggested = record["suggested_label"]
            record = {**record, "automatic_group": task.groups[0 if label else 1] if label is not None else "Needs review",
                      "suggested_group": task.groups[0 if suggested else 1] if suggested is not None else ""}
            # Preserve original strings in JSON, escape spreadsheet formulas in CSV.
            writer.writerow({field: "'" + value if isinstance(value := record.get(field), str)
                             and value.lstrip().startswith(("=", "+", "-", "@")) else value for field in fields})
        archive.writestr("results.csv", text.getvalue().encode("utf-8-sig"))
        archive.writestr("task.zip", task.export())
    return stream.getvalue()

"""Bounded in-memory image imports and source-isolated РГР dataset splits."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
import json
import stat
import warnings
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

from src.scene import Scene, VisualObject
from src.scene.renderer import draw_scene
from src.vision.attribute_cnn import SHAPE_CLASSES

MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGES = 2000
MAX_IMAGE_PIXELS = 16_000_000
MAX_TOTAL_PIXELS = 64_000_000
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class Sample:
    name: str
    image: np.ndarray
    label: str


def pixel_digest(image):
    array = np.asarray(image)
    return sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def validate_samples(samples):
    """Validate canonical RGB sources, rejecting contradictory repeated pixels."""
    samples = list(samples)
    if not samples:
        raise ValueError("The dataset contains no images.")
    if len(samples) > MAX_IMAGES:
        raise ValueError(f"Use at most {MAX_IMAGES} dataset images.")
    seen_pixels, seen_names, total_pixels = {}, set(), 0
    for sample in samples:
        if not isinstance(sample, Sample):
            raise ValueError("Each dataset entry must be a Sample.")
        if sample.label not in SHAPE_CLASSES:
            raise ValueError(f"Unknown class {sample.label!r}; use {', '.join(SHAPE_CLASSES)}.")
        if not isinstance(sample.name, str) or not sample.name or sample.name in seen_names:
            raise ValueError("Every dataset source needs a nonempty, unique filename.")
        seen_names.add(sample.name)
        array = np.asarray(sample.image)
        if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3
                or min(array.shape[:2]) < 8 or array.shape[0]*array.shape[1] > MAX_IMAGE_PIXELS):
            raise ValueError("Dataset images must be uint8 RGB, at least 8×8 and at most 16 megapixels.")
        total_pixels += array.shape[0]*array.shape[1]
        if total_pixels > MAX_TOTAL_PIXELS:
            raise ValueError("Decoded dataset exceeds the 64-megapixel total limit.")
        digest = pixel_digest(array)
        if digest in seen_pixels and seen_pixels[digest] != sample.label:
            raise ValueError("Identical pixels have conflicting class labels.")
        seen_pixels[digest] = sample.label
    return samples


def dataset_fingerprint(samples):
    samples = validate_samples(samples)
    identities = sorted((s.name, s.label, pixel_digest(s.image)) for s in samples)
    return sha256(json.dumps(identities, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_dataset_zip(data: bytes) -> list[Sample]:
    """Read class/image paths without extracting any archive member to disk."""
    if not isinstance(data, bytes) or len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Upload a ZIP no larger than 32 MB.")
    samples, names, expanded, total_pixels = [], set(), 0, 0
    try:
        with ZipFile(BytesIO(data)) as archive:
            if len(archive.infolist()) > MAX_IMAGES*3:
                raise ValueError("The ZIP contains too many entries.")
            for entry in archive.infolist():
                name = entry.filename
                original_name = entry.orig_filename
                path = PurePosixPath(name)
                if ("\\" in original_name or ":" in name or path.is_absolute() or ".." in path.parts
                        or "\x00" in name or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError("Archive contains an unsafe path or symbolic link.")
                if name in names:
                    raise ValueError("Archive contains duplicate filenames.")
                names.add(name)
                expanded += entry.file_size
                if expanded > MAX_EXPANDED_BYTES:
                    raise ValueError("Expanded ZIP exceeds the 128 MB limit.")
                if entry.is_dir():
                    continue
                if path.name.casefold() in {"readme", "readme.txt", "readme.md", "readme.rst"}:
                    continue
                parts = path.parts
                if parts and parts[0] == "dataset":
                    parts = parts[1:]
                if len(parts) != 2 or parts[0] not in SHAPE_CLASSES:
                    raise ValueError("Use class/image.png paths, optionally inside dataset/. Classes: "
                                     + ", ".join(SHAPE_CLASSES) + ".")
                if path.suffix.casefold() not in IMAGE_SUFFIXES:
                    raise ValueError(f"Unsupported dataset image format: {path.suffix}.")
                if entry.file_size > MAX_IMAGE_BYTES:
                    raise ValueError("An individual dataset image exceeds 12 MB.")
                if len(samples) >= MAX_IMAGES:
                    raise ValueError(f"Use at most {MAX_IMAGES} dataset images.")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(BytesIO(archive.read(entry))) as image:
                        if image.width*image.height > MAX_IMAGE_PIXELS or min(image.size) < 8:
                            raise ValueError("Dataset images must be 8×8 or larger and at most 16 megapixels.")
                        if getattr(image, "n_frames", 1) != 1:
                            raise ValueError("Animated dataset images are unsupported.")
                        total_pixels += image.width*image.height
                        if total_pixels > MAX_TOTAL_PIXELS:
                            raise ValueError("Decoded dataset exceeds the 64-megapixel total limit.")
                        from src.vision.pretrained import rgb_image
                        array = np.array(rgb_image(image))
                samples.append(Sample(name, array, parts[0]))
    except (BadZipFile, UnidentifiedImageError, OSError, RuntimeError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError(f"Could not read the dataset ZIP: {error}") from error
    return validate_samples(samples)


def split_samples(samples, seed=2026) -> dict[str, list[Sample]]:
    """Stratify 60/20/20 by unique source pixels before training augmentation.

    Exact duplicate sources stay in one split. There must be at least five
    independent source images in every represented class and two classes.
    """
    samples = validate_samples(samples)
    by_label = {}
    for sample in samples:
        by_label.setdefault(sample.label, {}).setdefault(pixel_digest(sample.image), []).append(sample)
    if len(by_label) < 2:
        raise ValueError("Provide at least two classes to fine-tune a classifier.")
    splits = {name: [] for name in ("train", "validation", "test")}
    rng = np.random.default_rng(seed)
    for label in sorted(by_label):
        groups = by_label[label]
        if len(groups) < 5:
            raise ValueError(f"Class {label!r} needs at least five unique source images; found {len(groups)}.")
        digests = sorted(groups)
        order = rng.permutation(len(digests))
        test_count = max(1, len(digests)//5)
        validation_count = max(1, len(digests)//5)
        for position, index in enumerate(order):
            split = ("test" if position < test_count else "validation"
                     if position < test_count+validation_count else "train")
            splits[split].extend(sorted(groups[digests[index]], key=lambda sample: sample.name))
    return splits


def evaluation_cohorts(samples):
    """Two disjoint, class-interleaved cohorts of up to five unique sources."""
    groups, seen = {}, set()
    for sample in samples:
        digest = pixel_digest(sample.image)
        if digest in seen:
            continue
        seen.add(digest)
        groups.setdefault(sample.label, []).append(sample)
    ordered = []
    for index in range(max(map(len, groups.values()), default=0)):
        for label in sorted(groups):
            if index < len(groups[label]):
                ordered.append(groups[label][index])
    return ordered[:5], ordered[5:10]


def demo_dataset(per_class=20, seed=2026) -> list[Sample]:
    """Generate reproducible synthetic sketches; these are never photographs."""
    if not isinstance(per_class, int) or not 5 <= per_class <= 300:
        raise ValueError("Synthetic demo needs 5–300 source images per class.")
    rng = np.random.default_rng(seed)
    samples = []
    for shape in SHAPE_CLASSES:
        for index in range(per_class):
            obj = VisualObject(0, shape, bool(rng.integers(2)), area=float(rng.uniform(.13, .24)),
                               angle=float(rng.uniform(-180, 180)))
            # Jitter each polygon vertex slightly to make independent, uneven
            # sketches without changing their semantic side counts.
            vertices = tuple((float(x+rng.normal(0, .003)), float(y+rng.normal(0, .003)))
                             for x, y in obj.vertices())
            obj = VisualObject(0, shape, obj.filled, area=obj.area, contour=vertices)
            image = draw_scene(Scene([obj]), size=192, stroke=int(rng.integers(2, 6)))
            array = np.asarray(image)
            ys, xs = np.where(array.mean(axis=2) < 180)
            pad = max(3, round(.08*max(np.ptp(xs), np.ptp(ys))))
            array = array[max(0, ys.min()-pad):min(192, ys.max()+pad+1),
                          max(0, xs.min()-pad):min(192, xs.max()+pad+1)].copy()
            if rng.random() < .5:
                array = np.asarray(Image.fromarray(array).filter(ImageFilter.GaussianBlur(float(rng.uniform(.1, .6)))))
            paper = rng.uniform(-12, 0) + np.linspace(-4, 4, array.shape[1])[None, :, None]
            noise = rng.normal(0, rng.uniform(1, 5), array.shape[:2])[..., None]
            array = np.uint8(np.clip(array.astype(float)+paper+noise, 0, 255))
            samples.append(Sample(f"synthetic/{shape}/{seed}_{index:04d}.png", array, shape))
    return samples

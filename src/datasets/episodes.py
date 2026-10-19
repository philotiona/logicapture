"""Dataset episodes with a pixel-only boundary between annotations and solvers.

The audit object deliberately retains split provenance and hidden concepts.
Only ``episode.prepare().input`` belongs at a solver boundary. Query truth,
paths and annotation metadata remain on the evaluation side of that boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from random import Random

import numpy as np
from PIL import Image


class DatasetError(ValueError):
    """An annotation violates the episode or file-access contract."""


def safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise DatasetError("An image reference must be a nonempty relative path")
    if "\\" in value or ":" in value or any(c in value for c in ('*', '?', '"', '|')):
        raise DatasetError("Windows roots, URI paths and ambiguous path syntax are unsupported")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts:
        raise DatasetError("Image paths must remain inside their configured root")
    if windows.is_reserved() or any(part.endswith((".", " ")) for part in posix.parts):
        raise DatasetError("Reserved or ambiguous Windows image filename")
    if not posix.parts or str(posix) == ".":
        raise DatasetError("Image path cannot refer to the root directory")
    return posix.as_posix()


def resolve_image(root: Path, relative_path: str, *, require_file=False) -> Path:
    root = Path(root).resolve()
    candidate = root.joinpath(*PurePosixPath(safe_relative_path(relative_path)).parts).resolve()
    if not candidate.is_relative_to(root):
        raise DatasetError("Image path or symlink escapes the configured root")
    if require_file and not candidate.is_file():
        raise FileNotFoundError(f"Dataset image is missing: {candidate}")
    return candidate


@dataclass(frozen=True)
class ImageRef:
    root: Path
    relative_path: str
    crop_bbox: tuple[int, int, int, int] | None = None
    coordinate_scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "relative_path", safe_relative_path(self.relative_path))
        resolve_image(self.root, self.relative_path)
        if not math.isfinite(self.coordinate_scale) or not 0 < self.coordinate_scale <= 10:
            raise DatasetError("Invalid annotation coordinate scale")
        if self.crop_bbox is not None:
            box = tuple(self.crop_bbox)
            if len(box) != 4 or any(type(v) is not int for v in box):
                raise DatasetError("Crop boxes require four integer pixel coordinates")
            x0, y0, x1, y1 = box
            if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
                raise DatasetError("Crop boxes must have a positive extent")
            object.__setattr__(self, "crop_bbox", box)

    @property
    def path(self):
        # Re-resolve at use time: a symlink may have changed since parsing.
        return resolve_image(self.root, self.relative_path, require_file=True)

    @property
    def source_identity(self):
        # Conservative source-image isolation, even when crop boxes differ.
        return str(resolve_image(self.root, self.relative_path)).casefold()

    def load_rgb(self):
        with Image.open(self.path) as image:
            if image.width*image.height > 80_000_000:
                raise DatasetError("Dataset image exceeds the 80-megapixel decoding limit")
            image = image.convert("RGB")
            if self.crop_bbox is not None:
                x0, y0, x1, y1 = [int(v*self.coordinate_scale) for v in self.crop_bbox]
                # Official HOI crops clamp to image bounds. No padding is added.
                x0, x1 = min(x0, image.width), min(x1, image.width)
                y0, y1 = min(y0, image.height), min(y1, image.height)
                if x0 >= x1 or y0 >= y1:
                    raise DatasetError(f"Annotation crop is empty for {self.relative_path}")
                image = image.crop((x0, y0, x1, y1))
            return np.array(image, dtype=np.uint8)


@dataclass(frozen=True)
class Query:
    image: ImageRef
    label: bool

    def __post_init__(self):
        if type(self.label) is not bool:
            raise DatasetError("Query truth must be Boolean")


@dataclass(frozen=True)
class SplitProvenance:
    source_url: str
    annotation_sha256: str
    source_split: str
    protocol: str = "first-six-support-last-query-per-group-v1"
    note: str = ""


@dataclass(frozen=True)
class SolverInput:
    """The only dataset information a solver receives: pixels and support roles."""
    positive: tuple[np.ndarray, ...]
    negative: tuple[np.ndarray, ...]
    queries: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PreparedEpisode:
    input: SolverInput
    query_labels: tuple[bool, ...]
    query_order: tuple[int, ...]


@dataclass(frozen=True)
class DatasetEpisode:
    dataset: str
    episode_id: str
    split: str
    support_positive: tuple[ImageRef, ...]
    support_negative: tuple[ImageRef, ...]
    queries: tuple[Query, ...]
    provenance: SplitProvenance
    hidden_concept: str | None = field(default=None, repr=False)

    def __post_init__(self):
        if not self.support_positive or not self.support_negative or not self.queries:
            raise DatasetError("An episode needs both support groups and held-out queries")
        refs = self.support_positive+self.support_negative+tuple(q.image for q in self.queries)
        identities = [ref.source_identity for ref in refs]
        if len(set(identities)) != len(identities):
            raise DatasetError("Repeated source image: support/query or within-group leakage")

    @property
    def images(self):
        return self.support_positive+self.support_negative+tuple(q.image for q in self.queries)

    def validate_files(self, *, check_content=False):
        hashes = set()
        for ref in self.images:
            path = ref.path
            if check_content:
                with path.open("rb") as stream:
                    digest = sha256()
                    for block in iter(lambda: stream.read(1024*1024), b""):
                        digest.update(block)
                value = digest.hexdigest()
                if value in hashes:
                    raise DatasetError("Byte-identical source images occur at different episode paths")
                hashes.add(value)

    def prepare(self, *, seed=0, check_content=True):
        self.validate_files(check_content=check_content)
        order = list(range(len(self.queries)))
        # Stable per-episode order avoids the official positive-query-first cue.
        token = sha256(f"{seed}:{self.dataset}:{self.split}:{self.episode_id}".encode()).digest()
        Random(int.from_bytes(token[:8], "big")).shuffle(order)
        return PreparedEpisode(SolverInput(
            tuple(ref.load_rgb() for ref in self.support_positive),
            tuple(ref.load_rgb() for ref in self.support_negative),
            tuple(self.queries[i].image.load_rgb() for i in order)),
            tuple(self.queries[i].label for i in order), tuple(order))


def audit_split_overlap(episodes, *, check_content=False):
    """Report source-image reuse across splits; never silently relabel splits."""
    owners, content_owners, content_paths, cache = {}, {}, {}, {}
    for episode in episodes:
        for ref in episode.images:
            identity = ref.source_identity
            owners.setdefault(identity, set()).add(episode.split)
            if check_content:
                if identity not in cache:
                    digest = sha256()
                    with ref.path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024*1024), b""):
                            digest.update(block)
                    cache[identity] = digest.hexdigest()
                value = cache[identity]
                content_owners.setdefault(value, set()).add(episode.split)
                content_paths.setdefault(value, set()).add(identity)
    overlap = {path: sorted(splits) for path, splits in owners.items() if len(splits) > 1}
    content = {digest: {"splits": sorted(splits), "paths": sorted(content_paths[digest])}
               for digest, splits in content_owners.items() if len(splits) > 1}
    return {"overlapping_source_images": len(overlap), "images": overlap,
            "byte_identical_across_splits": len(content) if check_content else None,
            "content_overlap": content if check_content else None}

"""Pixel-only orchestration for vocabulary-free episodic inference.

This is complementary to the executable symbolic solver. A fitted decision
boundary in visual features is never described as a recovered logical rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Sequence
import numpy as np

from .features import EpisodeSupport, FeatureBatch, fuse_feature_batches
from .episode import EpisodeClassifier, nearest_prototype


@dataclass
class GeneralAnalysis:
    predictions: list
    baseline_scores: list[float]
    diagnostics: dict
    encoder_id: str
    support_ids: tuple[str, ...]
    query_ids: tuple[str, ...]

    def to_dict(self):
        return {"predictions": [item.to_dict() for item in self.predictions],
                "baseline_scores": self.baseline_scores, "diagnostics": self.diagnostics,
                "encoder_id": self.encoder_id, "support_ids": self.support_ids, "query_ids": self.query_ids}


def validate_episode_images(positive, negative, queries=()):
    from src.vision.pretrained import rgb_image, image_digest
    if min(len(positive), len(negative)) < 2:
        raise ValueError("Use at least two support images in each group; four to six are preferable.")
    originals = list(positive)+list(negative)+list(queries)
    images = []
    hashes = []
    for image in originals:
        array = np.asarray(image)
        if array.ndim not in (2, 3) or min(array.shape[:2]) < 3 or array.dtype != np.uint8:
            raise ValueError("Each image must contain uint8 RGB or grayscale pixels, at least 3×3.")
        if array.size > 80_000_000*4: raise ValueError("Image exceeds decoding limit.")
        canonical = rgb_image(image)
        hashes.append(image_digest(canonical))
        images.append(np.array(canonical))
    count = len(positive)+len(negative)
    if set(hashes[:count]) & set(hashes[count:]):
        raise ValueError("An unseen image duplicates a support image. Use an independent drawing or photograph.")
    # Duplicate supports can be intentional (e.g. the empty-class classic
    # problem). Contradictory duplicates are diagnosed by the classifier.
    return images


@dataclass
class FittedVisualTask:
    """A support-only fit reusable across batches without learning from queries."""
    solver: object
    fitted: object
    diagnostics: dict
    encoder_id: str
    support_ids: tuple[str, ...]
    support_digests: frozenset[str]

    def predict(self, queries, *, image_ids=None, progress=None, remove_panel_ruling=None):
        from src.vision.pretrained import rgb_image, image_digest
        queries = list(queries)
        ids = tuple(image_ids) if image_ids is not None else tuple(f"Q{i+1:02d}" for i in range(len(queries)))
        if len(ids) != len(queries) or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("Provide one unique nonempty id per query image.")
        if set(ids) & set(self.support_ids):
            raise ValueError("Query ids overlap reference ids.")
        images = []
        for query in queries:
            array = np.asarray(query)
            if array.ndim not in (2, 3) or min(array.shape[:2]) < 3 or array.dtype != np.uint8:
                raise ValueError("Each query must contain uint8 RGB or grayscale pixels, at least 3×3.")
            if array.size > 80_000_000*4:
                raise ValueError("Image exceeds decoding limit.")
            image = np.array(rgb_image(query))
            if image_digest(image) in self.support_digests:
                raise ValueError("An unseen image duplicates a support image. Use an independent drawing or photograph.")
            images.append(image)
        structural = self.diagnostics["selected_pipeline"] != "visual_feature_classifier"
        if not images:
            predictions, scores = [], []
        elif structural:
            if remove_panel_ruling is not None and self.diagnostics["selected_pipeline"] == "executable_structural_rules":
                predictions = self.fitted.predict(images, ids, remove_panel_ruling=remove_panel_ruling)
            else:
                predictions = self.fitted.predict(images, ids)
            scores = []
        else:
            if progress:
                progress("Encoding new images with the fitted task")
            batch = self.solver.encode(images, ids)
            predictions = self.fitted.predict(batch)
            scores = nearest_prototype(self.fitted.support, batch).tolist()
        return GeneralAnalysis(predictions, scores, {**self.diagnostics, "heldout_queries": len(images)},
                               self.encoder_id, self.support_ids, ids)


class GeneralVisualSolver:
    def __init__(self, semantic_encoder=None, structural_encoder=None, classifier=None, *, structural_first=False):
        if semantic_encoder is None and structural_encoder is None and not structural_first:
            raise ValueError("Provide at least one visual encoder.")
        self.semantic_encoder, self.structural_encoder = semantic_encoder, structural_encoder
        self.classifier = classifier or EpisodeClassifier()
        # Explicit opt-in keeps published feature-only benchmark ablations
        # reproducible. The application enables this automatic route by default.
        self.structural_first = structural_first

    def encode(self, images: Sequence, image_ids):
        batches = {}
        if self.semantic_encoder is not None:
            batches["semantic"] = self.semantic_encoder.encode(images, image_ids)
        if self.structural_encoder is not None:
            encoded = self.structural_encoder.encode(images)
            batches["structural"] = (FeatureBatch(tuple(image_ids), encoded.global_features, encoded.patch_features,
                                                  encoded.encoder_id, encoded.metadata, encoded.extra_views)
                                     if isinstance(encoded, FeatureBatch) else encoded.as_feature_batch(image_ids))
        return next(iter(batches.values())) if len(batches) == 1 else fuse_feature_batches(batches)

    def fit(self, positive, negative, *, progress=None):
        """Freeze a rule or feature classifier using reference images only."""
        from src.vision.pretrained import image_digest
        images = validate_episode_images(positive, negative)
        support_count = len(positive)+len(negative)
        ids = tuple([f"A{i+1:02d}" for i in range(len(positive))] +
                    [f"B{i+1:02d}" for i in range(len(negative))])
        digests = frozenset(image_digest(image) for image in images)
        structural_attempt = {"accepted": False, "reason": "disabled_for_feature_baseline"}
        if self.structural_first:
            from .drawing import fit_drawing_rules
            fitted_rules, structural_attempt, pipeline = fit_drawing_rules(
                images[:len(positive)], images[len(positive):support_count], progress=progress)
            if fitted_rules is not None:
                diagnostics = {"input": "pixels and support membership only", "local_only": True,
                               "query_labels_used": False,
                               "architecture": "traced drawing structure + executable symbolic rules",
                               "selected_pipeline": pipeline,
                               "structural_attempt": structural_attempt,
                               "logical_explanation": structural_attempt["candidates"][0]["text"],
                               "warning": "Support validation does not prove that this is the intended rule."}
                return FittedVisualTask(self, fitted_rules, diagnostics, "adaptive-structure-dsl-v1", ids, digests)
        if self.semantic_encoder is None and self.structural_encoder is None:
            raise ValueError("No validated structural rule; provide a visual encoder for feature fallback. "
                             + structural_attempt["reason"])
        if progress: progress("Encoding pixels locally")
        batch = self.encode(images, ids)
        support = EpisodeSupport(batch.subset(range(support_count)),
                                 (True,)*len(positive)+(False,)*len(negative))
        if progress: progress("Selecting representations and learner using support-only validation")
        fitted = self.classifier.fit(support)
        diagnostics = {**fitted.diagnostics, "input": "pixels and support membership only",
                       "selected_pipeline": "visual_feature_classifier", "structural_attempt": structural_attempt,
                       "representation_views": list(batch.views()),
                       "local_only": True,
                       "architecture": "frozen visual encoders + support-adapted statistical classifier",
                       "logical_explanation": None,
                       "warning": "A visual classifier is not a proof of the intended Bongard rule."}
        return FittedVisualTask(self, fitted, diagnostics, batch.encoder_id, ids, digests)

    def solve(self, positive, negative, queries=(), *, progress=None):
        # Validate cross-split identity before doing potentially costly fitting.
        validate_episode_images(positive, negative, queries)
        return self.fit(positive, negative, progress=progress).predict(queries, progress=progress)

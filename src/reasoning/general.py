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

    def solve(self, positive, negative, queries=(), *, progress=None):
        images = validate_episode_images(positive, negative, queries)
        support_count = len(positive)+len(negative)
        ids = tuple([f"A{i+1:02d}" for i in range(len(positive))] +
                    [f"B{i+1:02d}" for i in range(len(negative))] +
                    [f"Q{i+1:02d}" for i in range(len(queries))])
        structural_attempt = {"accepted": False, "reason": "disabled_for_feature_baseline"}
        if self.structural_first:
            from .structural import fit_structural_support
            fitted_rules, structural_attempt = fit_structural_support(
                images[:len(positive)], images[len(positive):support_count], progress=progress)
            if fitted_rules is not None:
                predictions = fitted_rules.predict(images[support_count:], ids[support_count:])
                diagnostics = {"input": "pixels and support membership only", "local_only": True,
                               "heldout_queries": len(queries), "query_labels_used": False,
                               "architecture": "traced drawing structure + executable symbolic rules",
                               "selected_pipeline": "executable_structural_rules",
                               "structural_attempt": structural_attempt,
                               "logical_explanation": structural_attempt["candidates"][0]["text"],
                               "warning": "Support validation does not prove that this is the intended rule."}
                return GeneralAnalysis(predictions, [], diagnostics, "adaptive-structure-dsl-v1",
                                       ids[:support_count], ids[support_count:])
        if self.semantic_encoder is None and self.structural_encoder is None:
            raise ValueError("No validated structural rule; provide a visual encoder for feature fallback. "
                             + structural_attempt["reason"])
        if progress: progress("Encoding pixels locally")
        batch = self.encode(images, ids)
        support = EpisodeSupport(batch.subset(range(support_count)),
                                 (True,)*len(positive)+(False,)*len(negative))
        if progress: progress("Selecting representations and learner using support-only validation")
        fitted = self.classifier.fit(support)
        if len(queries):
            query = batch.subset(range(support_count, len(images)))
            predictions = fitted.predict(query)
            scores = nearest_prototype(support, query).tolist()
        else: predictions, scores = [], []
        diagnostics = {**fitted.diagnostics, "input": "pixels and support membership only",
                       "selected_pipeline": "visual_feature_classifier", "structural_attempt": structural_attempt,
                       "representation_views": list(batch.views()),
                       "heldout_queries": len(queries), "local_only": True,
                       "architecture": "frozen visual encoders + support-adapted statistical classifier",
                       "logical_explanation": None,
                       "warning": "A visual classifier is not a proof of the intended Bongard rule."}
        return GeneralAnalysis(predictions, scores, diagnostics, batch.encoder_id,
                               ids[:support_count], ids[support_count:])

"""Fit a visible relationship on references before sorting independent photos."""
from dataclasses import dataclass

from src.vision.pretrained import image_digest
from .concepts import check_visible_condition, propose_visible_condition
from .episode import EpisodePrediction
from .general import GeneralAnalysis


@dataclass
class FittedPhotoRelation:
    backend: object
    condition: str | None
    diagnostics: dict
    support_ids: tuple
    support_digests: frozenset

    @property
    def encoder_id(self):
        return self.backend.model_id

    def predict(self, images, *, image_ids, progress=None):
        predictions = []
        for index, (image_id, image) in enumerate(zip(image_ids, images, strict=True), start=1):
            if progress:
                progress(f"Checking the visual condition on image {index} of {len(images)}")
            if self.condition is None:
                label, evidence = None, ""
                reasons = ("unverified_photo_relation",)
            else:
                judgment = check_visible_condition(self.backend, self.condition, image)
                label, evidence = judgment.label, judgment.evidence
                reasons = ("unclear_photo_relation",) if label is None else ()
            predictions.append(EpisodePrediction(image_id, label, label, None, None, reasons, (),
                                                 {"visible_evidence": evidence}))
        return GeneralAnalysis(predictions, [], self.diagnostics, self.encoder_id, self.support_ids, tuple(image_ids))


def fit_photo_relation(task, backend, *, progress=None):
    references = task.positive + task.negative
    ids = tuple(f"{prefix}{i:02d}" for prefix, group in zip("AB", (task.positive, task.negative))
                for i in range(1, len(group)+1))
    condition = propose_visible_condition(backend, task.positive, task.negative, progress=progress)
    checks = []
    if condition:
        for index, (image_id, image) in enumerate(zip(ids, references), start=1):
            if progress:
                progress(f"Verifying the condition on reference {index} of {len(references)}")
            judgment = check_visible_condition(backend, condition, image)
            expected = image_id.startswith("A")
            checks.append({"id": image_id, "expected": expected, "observed": judgment.label,
                           "matches": judgment.label is expected, "visible_evidence": judgment.evidence})
    verified = bool(checks) and all(check["matches"] for check in checks)
    diagnostics = {
        "selected_pipeline": "photo_relation", "proposed_condition": condition, "condition_verified": verified,
        "reference_checks": checks, "support_consistency_is_heldout_accuracy": False,
        "query_labels_used": False, "local_only": True,
    }
    return FittedPhotoRelation(backend, condition if verified else None, diagnostics, ids,
                               frozenset(image_digest(image) for image in references))

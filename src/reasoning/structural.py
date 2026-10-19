"""Support-only routing to executable rules before visual feature fallback.

This is a bounded drawing specialist, not a universal concept recognizer. The
same fixed grammar is fitted afresh in every validation fold. Neither query
pixels nor their answers select a representation, rule, or routing decision.
"""
from dataclasses import asdict, dataclass
from itertools import product

import cv2
import numpy as np

from src.dsl.grammar import enumerate_rules
from src.dsl.predictions import assessment_matrix
from src.dsl.verbalizer import verbalize
from src.scene.strokes import EVIDENCE_THRESHOLD
from src.vision import extract_scene
from .episode import EpisodePrediction


def _drawing(image):
    """Conservative applicability gate; it does not identify a Bongard concept."""
    if min(image.shape[:2]) < 24:
        return False
    pixels = image.astype(np.int16)
    gray = pixels.mean(axis=2)
    return (np.mean(pixels.max(axis=2)-pixels.min(axis=2) > 30) < .02
            and np.mean(gray > 220) > .55
            and np.mean((gray > 50) & (gray < 220)) < .20)


def _read(image):
    if not _drawing(image):
        return None
    # Bound skeletonization cost on large uploads. Preserve aspect ratio.
    h, w = image.shape[:2]
    if max(h, w) > 768:
        image = cv2.resize(image, (round(w*768/max(h, w)), round(h*768/max(h, w))),
                           interpolation=cv2.INTER_AREA)
    # Dense dot fields are handled by measured relations. Avoid tracing
    # hundreds of disconnected marks before discovering the object budget.
    from src.vision.adaptive_perception import ink_mask, remove_ruling
    mask,_ = remove_ruling(ink_mask(image))
    _,_,stats,_ = cv2.connectedComponentsWithStats(mask,8)
    if sum(area >= max(12,.0009*mask.size) for area in stats[1:,4]) > 24:
        return None
    scene = extract_scene(image, representation="adaptive", remove_panel_ruling=True)
    if (len(scene.objects) > 24
            or any(obj.confidence < EVIDENCE_THRESHOLD or obj.kind == "unknown" for obj in scene.objects)
            or any("touches the panel boundary" in warning for warning in scene.warnings)):
        return None
    return scene


def _cost(rule):
    return (rule.complexity + int(rule.op in ("exactly", "atleast", "compare"))
            + sum(_cost(child)-child.complexity for child in rule.children))


def _minimal_consistent(matrix, labels, indices, costs):
    valid = np.flatnonzero(np.all(matrix[:, indices] == labels[indices], axis=1))
    return valid[costs[valid] == costs[valid].min()] if len(valid) else valid


def _consensus(values):
    unique = set(values)
    return bool(next(iter(unique))) if len(unique) == 1 and -1 not in unique else None


def _evidence(scene):
    if scene is None:
        return {"perception": "unresolved"}
    return {"objects": [{"id": obj.id, "kind": obj.kind, "shape": obj.shape,
                          **({"topology": obj.path.topology,
                              "arrow_count": len(obj.path.arrows),
                              "along_path_flow": obj.path.arrow_relation,
                              "direction_confidence": obj.path.direction_confidence}
                             if obj.path is not None else {})} for obj in scene.objects],
            "warnings": scene.warnings}


@dataclass
class StructuralFit:
    rules: tuple
    diagnostics: dict

    def predict(self, images, image_ids):
        predictions = []
        for image, image_id in zip(images, image_ids):
            scene = _read(image)
            votes = [-1 if (v := rule.assess(scene)) is None else int(v)
                     for rule in self.rules] if scene is not None else [-1]*len(self.rules)
            label = _consensus(votes)
            reasons = (() if label is not None else
                       ("unresolved_structural_evidence",) if -1 in votes else ("structural_rules_disagree",))
            predictions.append(EpisodePrediction(
                image_id, label, label, 0. if label is None else 1. if label else -1., None,
                reasons, (), {"rule_votes": votes, "measured_evidence": _evidence(scene)}))
        return predictions


def fit_structural_support(positive, negative, *, progress=None):
    """Return a validated frozen fit, or an explicit reason to try features.

    All equally simple, perfectly fitting programs must agree on every omitted
    support in every leave-one-per-group fold. This is an empirical check, not
    a guarantee that the intended rule is represented or uniquely recovered.
    """
    attempt = {"accepted": False, "query_images_used_in_selection": False}
    if min(len(positive), len(negative)) < 3:
        return None, {**attempt, "reason": "needs_three_supports_per_group"}
    if max(len(positive), len(negative)) > 8:
        return None, {**attempt, "reason": "structural_validation_budget_exceeded"}
    if not all(_drawing(image) for image in list(positive)+list(negative)):
        return None, {**attempt, "reason": "outside_monochrome_drawing_domain"}
    if progress:
        progress("Tracing drawing structure and validating executable rules on support images")
    scenes = [_read(image) for image in list(positive)+list(negative)]
    if any(scene is None for scene in scenes):
        return None, {**attempt, "reason": "unresolved_support_perception"}
    rules = enumerate_rules(depth=2, paths=True)
    matrix = assessment_matrix(rules, scenes)
    labels = np.array([True]*len(positive)+[False]*len(negative))
    costs = np.array([_cost(rule) for rule in rules])
    indices = np.arange(len(scenes))
    selected = _minimal_consistent(matrix, labels, indices, costs)
    if not len(selected):
        return None, {**attempt, "reason": "no_consistent_structural_rule"}
    correct, total, folds = 0, 0, 0
    for a, b in product(range(len(positive)), range(len(positive), len(scenes))):
        train = indices[(indices != a) & (indices != b)]
        candidates = _minimal_consistent(matrix, labels, train, costs)
        folds += 1
        for heldout in (a, b):
            vote = _consensus(matrix[candidates, heldout])
            correct += vote is not None and vote == bool(labels[heldout])
            total += 1
    attempt.update(validation_correct=int(correct), validation_total=total,
                   validation_folds=folds, validation_accuracy=correct/total,
                   validation_method="leave_one_per_group_refit_minimum_cost_rules")
    if correct != total:
        return None, {**attempt, "reason": "structural_support_validation_failed"}
    chosen = tuple(rules[index] for index in selected)
    attempt.update(accepted=True, reason="validated_executable_rules", support_correct=len(scenes),
                   support_total=len(scenes), candidate_count=len(chosen),
                   candidates=[{"text": verbalize(rule), "program": asdict(rule)} for rule in chosen],
                   support_evidence=[_evidence(scene) for scene in scenes])
    return StructuralFit(chosen, attempt), attempt

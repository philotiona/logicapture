"""Pixel-to-rule checks for a supplied path problem, with abstention explicit.

Expected labels and arrow annotations are read only *after* perception. They
score the output; neither the extractor nor the rule receives those annotations.
This is a regression check for one screenshot, not a Bongard benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def selective_metrics(predictions, labels):
    """Separate coverage from correctness instead of counting unknown as False."""
    if len(predictions) != len(labels) or not labels:
        raise ValueError("Predictions and labels must have equal nonzero lengths")
    if any(p is not None and not isinstance(p, (bool, np.bool_)) for p in predictions):
        raise TypeError("A prediction must be Boolean or None (abstention)")
    answered = [i for i, value in enumerate(predictions) if value is not None]
    correct = sum(bool(predictions[i]) == bool(labels[i]) for i in answered)
    return {"total": len(labels), "answered": len(answered),
            "abstained": len(labels)-len(answered), "correct": correct,
            "coverage": len(answered)/len(labels),
            "selective_accuracy": correct/len(answered) if answered else None,
            "overall_correct_fraction": correct/len(labels)}


def pixel_variants(image):
    """Invariant transformations for relative arrow flow, not screen direction."""
    return {
        "original": image.copy(),
        "rotate_90": np.rot90(image).copy(),
        "rotate_180": np.rot90(image, 2).copy(),
        "rotate_270": np.rot90(image, 3).copy(),
        "mirror_horizontal": np.fliplr(image).copy(),
        "mirror_vertical": np.flipud(image).copy(),
        "translate_with_padding": cv2.copyMakeBorder(image, 8, 12, 18, 9,
                                                      cv2.BORDER_CONSTANT, value=(255, 255, 255)),
        "resize_150pct": cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC),
        "resize_75pct": cv2.resize(image, None, fx=.75, fy=.75, interpolation=cv2.INTER_AREA),
    }


def _describe_path(scene):
    paths = [o.path for o in scene.objects if getattr(o, "path", None) is not None]
    if len(paths) != 1:
        return {"path_count": len(paths), "relation": None, "arrow_count": None}
    path = paths[0]
    return {"path_count": 1, "relation": path.arrow_relation,
            "arrow_count": len(path.arrows), "topology": path.topology,
            "confidence": path.confidence,
            "endpoint_count": path.graph_count("endpoint"),
            "junction_count": path.graph_count("junction"),
            "cycle_count": path.graph_count("cycle"),
            "arrows": [{"direction": a.direction, "confidence": a.confidence,
                        "arc_position": a.arc_position, "tip": a.tip, "tail": a.tail}
                       for a in path.arrows]}


def _fit(examples, fit):
    result = fit(examples)
    # A partial or unknown-covered explanation is not a recovered rule.
    if not result.ranked or not result.perfect_count:
        return None, result
    best = result.ranked[0]
    if getattr(best, "coverage", 1.0) < 1.0 or best.correct != best.total:
        return None, result
    return best.rule, result


def evaluate_fixture(manifest_path="data/fixtures/arrow_paths/manifest.json", *,
                     extract=None, fit=None):
    """Fit on every held-out pair and transform pixels before re-extracting."""
    if extract is None:
        from src.vision import extract_scene
        extract = lambda image: extract_scene(image, representation="adaptive")
    if fit is None:
        from src.synthesis import search
        fit = lambda examples: search(examples, paths=True)
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest["items"]
    # File paths load pixels only. No fixture identity, A/B label, or truth
    # annotation crosses the perception interface.
    images = []
    for item in items:
        image = cv2.imread(str(manifest_path.parent/item["file"]))
        if image is None:
            raise FileNotFoundError(item["file"])
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    scenes = [extract(image) for image in images]
    labels = [bool(item["label"]) for item in items]
    rows = []
    for item, scene in zip(items, scenes):
        measured = _describe_path(scene)
        measured.update(file=item["file"], expected_relation=item["expected_arrow_relation"],
                        relation_correct=measured["relation"] == item["expected_arrow_relation"],
                        expected_arrow_count=item["expected_arrows"])
        rows.append(measured)
    rule, result = _fit(list(zip(scenes, labels)), fit)
    original_predictions = [rule.assess(scene) for scene in scenes] if rule else [None]*len(items)
    transforms = []
    variants = [pixel_variants(image) for image in images]
    for name in variants[0]:
        transformed_scenes = [extract(variant[name]) for variant in variants]
        # Assess using the original recovered concept: refitting could hide a
        # transformation-specific shortcut or a swapped relation.
        predictions = [rule.assess(scene) for scene in transformed_scenes] if rule else [None]*len(items)
        transforms.append({"name": name, **selective_metrics(predictions, labels),
                           "predictions": predictions,
                           "paths": [_describe_path(scene) for scene in transformed_scenes]})
    group_a = [i for i, label in enumerate(labels) if label]
    group_b = [i for i, label in enumerate(labels) if not label]
    if len(group_a) != len(group_b):
        raise ValueError("Pairwise holdout requires equally sized groups")
    folds, heldout_predictions, heldout_labels = [], [], []
    for a, b in zip(group_a, group_b):
        held = {a, b}
        train = [(s, y) for i, (s, y) in enumerate(zip(scenes, labels)) if i not in held]
        fold_rule, fold_result = _fit(train, fit)
        predictions = [fold_rule.assess(scenes[i]) for i in (a, b)] if fold_rule else [None, None]
        truth = [labels[i] for i in (a, b)]
        folds.append({"held_out": [items[a]["file"], items[b]["file"]],
                      "rule": fold_rule.key if fold_rule else None,
                      "perfect_hypotheses": fold_result.perfect_count,
                      "predictions": predictions, **selective_metrics(predictions, truth)})
        heldout_predictions.extend(predictions)
        heldout_labels.extend(truth)
    # Reordering and label inversion exercise concept induction, not perception.
    reversed_rule, _ = _fit(list(reversed(list(zip(scenes, labels)))), fit)
    inverted_labels = [not y for y in labels]
    inverted_rule, _ = _fit(list(zip(scenes, inverted_labels)), fit)
    ordered_predictions = [reversed_rule.assess(s) for s in scenes] if reversed_rule else [None]*len(items)
    inverted_predictions = [inverted_rule.assess(s) for s in scenes] if inverted_rule else [None]*len(items)
    blank_scenes = [extract(np.full_like(image, 255)) for image in images]
    blank_rule, blank_result = _fit(list(zip(blank_scenes, labels)), fit)
    # Identical observed evidence assigned to both groups must not yield a
    # separator, even if a noisy-mode ranking offers a best partial candidate.
    contradictory_rule, contradictory_result = _fit([(scenes[0], True), (scenes[0], False)], fit)
    controls = [{"name": "blank_panels", "accepted_rule": blank_rule.key if blank_rule else None,
                 "perfect_hypotheses": blank_result.perfect_count},
                {"name": "identical_pixels_in_both_groups",
                 "accepted_rule": contradictory_rule.key if contradictory_rule else None,
                 "perfect_hypotheses": contradictory_result.perfect_count}]
    return {"source_manifest": str(manifest_path), "provenance": manifest.get("provenance"),
            "scope": "One supplied screenshot problem; transforms are dependent regression cases, not new problems.",
            "annotation_use": "Expected labels score results; arrow truth is never supplied to perception or search.",
            "original": {**selective_metrics(original_predictions, labels),
                         "rule": rule.key if rule else None, "perfect_hypotheses": result.perfect_count,
                         "panels": rows},
            "heldout": {**selective_metrics(heldout_predictions, heldout_labels), "folds": folds},
            "transforms": transforms,
            "reordered": selective_metrics(ordered_predictions, labels),
            "labels_swapped": selective_metrics(inverted_predictions, inverted_labels),
            "negative_controls": controls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/fixtures/arrow_paths/manifest.json")
    parser.add_argument("--output", default="data/regressions/structural_regression.json")
    args = parser.parse_args()
    report = evaluate_fixture(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"original": {k: v for k, v in report["original"].items() if k != "panels"},
                      "heldout": {k: v for k, v in report["heldout"].items() if k != "folds"},
                      "transforms": [{k: v for k, v in row.items() if k not in ("paths", "predictions")}
                                     for row in report["transforms"]],
                      "reordered": report["reordered"], "labels_swapped": report["labels_swapped"],
                      "negative_controls": report["negative_controls"]}, indent=2))
    checks = [report["original"], report["heldout"], report["reordered"], report["labels_swapped"],
              *report["transforms"]]
    if any(check["overall_correct_fraction"] < 1 for check in checks) or any(
            row["accepted_rule"] is not None for row in report["negative_controls"]):
        raise SystemExit(1)

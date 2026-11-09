"""Regression for the default app bypassing its working path-reasoning engine."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.reasoning.general import GeneralVisualSolver
from src.reasoning.features import FeatureBatch
from src.reasoning import structural
from src.dsl.nodes import Exists, Count, Predicate
from src.active.path_generation import path_object
from src.scene import Scene
from src.scene.renderer import draw_scene


@pytest.fixture(scope="module")
def arrows():
    folder = Path(__file__).resolve().parents[1]/"data/fixtures/arrow_paths"
    return [[np.array(Image.open(folder/f"{group}{i:02d}.png").convert("RGB"))
             for i in range(1, 7)] for group in "AB"]


class NoFeatureFallback:
    def encode(self, *args, **kwargs):
        raise AssertionError("Validated path rules must not be replaced by image-feature classification")


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
def test_original_arrow_pixels_route_without_group_or_filename_assumptions(arrows, invert, reorder):
    a, b = arrows
    positive, negative = (b, a) if invert else (a, b)
    p, n = positive[:-1], negative[:-1]
    if reorder:
        p, n = p[::-1], [n[i] for i in (2, 4, 1, 0, 3)]
    result = GeneralVisualSolver(NoFeatureFallback(), structural_first=True).solve(p, n, [a[-1], b[-1]])
    assert [prediction.label for prediction in result.predictions] == [not invert, invert]
    assert result.diagnostics["selected_pipeline"] == "executable_structural_rules"
    assert "arrows point" in result.diagnostics["logical_explanation"]
    check = result.diagnostics["structural_attempt"]
    assert check["validation_correct"] == check["validation_total"] == 50
    assert check["query_images_used_in_selection"] is False
    assert result.diagnostics["query_labels_used"] is False
    for prediction, flow in zip(result.predictions, ("opposed", "same")):
        obj = prediction.diagnostics["measured_evidence"]["objects"][0]
        assert obj["along_path_flow"] == flow
        assert obj["kind"] == "path" and obj["shape"] == "unknown"
        assert prediction.probability is None


def test_frozen_rule_ignores_query_batch_and_abstains_on_missing_evidence(arrows):
    a, b = arrows
    solver = GeneralVisualSolver(NoFeatureFallback(), structural_first=True)
    one = solver.solve(a[:-1], b[:-1], [a[-1]])
    plain_curve = np.array(draw_scene(Scene([path_object("wave", ())])))
    blank = np.full((160, 160, 3), 255, np.uint8)
    many = solver.solve(a[:-1], b[:-1], [blank, b[-1], a[-1], plain_curve])
    assert one.diagnostics == many.diagnostics | {"heldout_queries": 1}
    assert one.predictions[0].label is many.predictions[2].label is True
    assert many.predictions[0].label is None  # Exists/forall alternatives disagree on empty scenes.
    assert many.predictions[3].label is None  # Absence of measured direction is not a negative.
    assert "unresolved_structural_evidence" in many.predictions[3].abstention_reasons


def test_cross_validation_refits_instead_of_reusing_the_full_support_rule(monkeypatch):
    # A higher-cost rule fits all supports, but a simpler shortcut wins when
    # A01 is withheld. Reusing the full-data rule would falsely report 100%.
    monkeypatch.setattr(structural, "_read", lambda image, **kwargs: Scene())
    monkeypatch.setattr(structural, "enumerate_rules", lambda **kwargs: [Exists(Predicate()), Count(1)])
    matrix = np.array([[0, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]])
    monkeypatch.setattr(structural, "assessment_matrix", lambda *args: matrix)
    white = np.full((40, 40, 3), 255, np.uint8)
    fit, diagnostic = structural.fit_structural_support([white]*3, [white]*3)
    assert fit is None
    assert diagnostic["reason"] == "structural_support_validation_failed"
    assert diagnostic["validation_correct"] < diagnostic["validation_total"]


def _standalone_rectangle():
    image = np.full((200, 200, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (180, 180), (0, 0, 0), 1)
    return image


def test_standalone_rectangle_is_never_silently_erased_into_an_empty_scene(arrows):
    # Without frame provenance, the ruling heuristic cannot tell this outline
    # from a panel frame. Perception must keep it or abstain, never return [].
    image = _standalone_rectangle()
    scene = structural._read(image)
    assert scene is None or len(scene.objects) >= 1
    a, b = arrows
    fit, _ = structural.fit_structural_support(a[:-1], b[:-1])
    if scene is None:
        prediction = fit.predict([image], ["rectangle"])[0]
        assert prediction.label is None
        assert prediction.diagnostics["measured_evidence"] == {"perception": "unresolved"}
        assert "unresolved_structural_evidence" in prediction.abstention_reasons
        _, diagnostic = structural.fit_structural_support([image]+a[:2], b[:3])
        assert diagnostic["accepted"] is False
        assert diagnostic["reason"] == "unresolved_support_perception"


def test_unframed_rule_uses_the_outer_shape_instead_of_fill_or_size_shortcuts():
    from src.reasoning.drawing import fit_drawing_rules

    def panel(rectangular, seed, *, challenge=False):
        rng = np.random.default_rng(seed)
        image = np.full((240, 240, 3), 255, np.uint8)
        ink = (0, 0, 0)
        radius = 120 - int(rng.integers(19, 33))
        x, y = map(int, rng.integers(114, 127, 2))
        thickness = int(rng.integers(1, 4))
        if rectangular:
            cv2.rectangle(image, (x-radius, y-radius), (x+radius, y+radius), ink, thickness)
        elif not challenge:
            cv2.circle(image, (x, y), radius, ink, 4)
        for x in (90, 151)[:1+seed % 2]:
            center = (x, int(rng.integers(101, 143)))
            cv2.circle(image, center, int(rng.integers(10, 19)), ink,
                       3 if rectangular and challenge else -1)
        return image

    positive = [panel(True, seed) for seed in range(4)]
    negative = [panel(False, seed) for seed in range(10, 14)]
    fit, diagnostic, route = fit_drawing_rules(positive, negative, remove_panel_ruling=False)
    assert route == "executable_structural_rules" and diagnostic["accepted"]
    assert any(item["shape"] == "quadrilateral" for item in diagnostic["support_evidence"][0]["objects"])

    # Fresh variations include hollow inner circles and small filled negatives,
    # so removing the outer rectangles cannot be rescued by fill or size.
    truth = [index % 2 == 0 for index in range(12)]
    queries = [panel(label, index+100, challenge=index % 4 < 2) for index, label in enumerate(truth)]
    decisions = fit.predict(queries, [f"Q{index+1:02d}" for index in range(len(queries))])
    assert [prediction.label for prediction in decisions] == truth


def test_outside_drawing_domain_falls_back_without_inventing_a_rule():
    class Encoder:
        calls = 0
        def encode(self, images, image_ids):
            self.calls += 1
            features = np.array([[image.mean()/255, 1.] for image in images])
            return FeatureBatch(tuple(image_ids), features, encoder_id="stub-feature-fallback")
    encoder = Encoder()
    images = [np.full((32, 32, 3), value, np.uint8) for value in (10, 20, 30, 180, 190, 200, 40)]
    result = GeneralVisualSolver(encoder, structural_first=True).solve(images[:3], images[3:6], images[6:])
    # The reusable fit encodes references first, then encodes new queries using
    # the frozen fit. Neither query pixels nor labels select its parameters.
    assert encoder.calls == 2
    assert result.diagnostics["selected_pipeline"] == "visual_feature_classifier"
    assert result.diagnostics["structural_attempt"]["reason"] == "outside_monochrome_drawing_domain"
    assert result.diagnostics["logical_explanation"] is None

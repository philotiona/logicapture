import cv2
import numpy as np
import pytest

from src.reasoning.examples import _boundary_drawing, geometric_demo, generate_checked_examples
from src.reasoning.general import GeneralVisualSolver
from src.reasoning.measurement_rules import MeasurementRule
from src.vision.observations import observations


@pytest.mark.parametrize("filled", [False, True])
@pytest.mark.parametrize("corners", [3, 4, 5, 7])
def test_simplified_convexity_survives_small_rasterized_polygons(filled, corners):
    image = _boundary_drawing(np.random.default_rng(14), size=96, concave=False,
                              elongated=False, filled=filled, corners=corners)
    values = observations(image)
    assert values["minimum_boundary_convexity"] > .98
    concave = _boundary_drawing(np.random.default_rng(14), size=160, concave=True,
                                elongated=False, filled=filled, corners=corners)
    assert observations(concave)["minimum_boundary_convexity"] < .9


@pytest.mark.parametrize("angle", [0, 17, 43, 90, 133])
def test_elongation_is_independent_of_screen_axes(angle):
    image = np.full((256, 256, 3), 255, np.uint8)
    points = cv2.boxPoints(((128, 128), (130, 26), angle)).round().astype(np.int32)
    cv2.fillPoly(image, [points], (0, 0, 0))
    values = observations(image)
    assert 4.6 < values["median_elongation"] < 5.2
    assert values["minimum_elongation"] == values["maximum_elongation"]


def test_missing_or_clipped_boundaries_do_not_invent_geometry():
    image = np.full((100, 100, 3), 255, np.uint8)
    assert observations(image)["median_boundary_convexity"] is None
    cv2.rectangle(image, (0, 25), (60, 60), (0, 0, 0), -1)
    values = observations(image)
    assert values["minimum_boundary_convexity"] is None
    assert values["median_elongation"] is None


class NoFallback:
    def encode(self, *args, **kwargs):
        raise AssertionError("The generated geometry should recover an executable drawing rule")


@pytest.mark.parametrize("concept", ["convexity", "elongation"])
@pytest.mark.parametrize("inverted", [False, True])
def test_geometry_rules_are_inferred_from_support_pixels_and_generate_verified_examples(concept, inverted):
    a, b = geometric_demo(concept)
    positive, negative = (b, a) if inverted else (a, b)
    result = GeneralVisualSolver(NoFallback(), structural_first=True).solve(
        positive[:-1], negative[:-1], [a[-1], b[-1]])
    assert result.diagnostics["selected_pipeline"] == "executable_measurement_rules"
    assert [prediction.label for prediction in result.predictions] == [not inverted, inverted]
    attempt = result.diagnostics["structural_attempt"]
    assert attempt["query_images_used_in_selection"] is False
    assert attempt["validation_correct"] == attempt["validation_total"]
    assert attempt["validation_is_independent_accuracy"] is False
    programs = [item["program"] for item in attempt["candidates"]]
    assert any(concept in program["feature"] for program in programs)
    examples = generate_checked_examples(result.diagnostics, attempts=128)
    assert [example.label for example in examples] == [True, False]
    for example in examples:
        measured = observations(example.image)
        assert example.verification["pixel_reread"]
        assert not any(np.array_equal(example.image, support) for support in a+b)
        for program in programs:
            rule = MeasurementRule(program["feature"], program["threshold"], program["positive_below"])
            assert rule.assess(measured) is example.label


def test_demo_is_reproducible_and_rejects_unsupported_sizes():
    first, second = geometric_demo("convexity"), geometric_demo("convexity")
    assert all(np.array_equal(a, b) for left, right in zip(first, second) for a, b in zip(left, right))
    with pytest.raises(ValueError):
        geometric_demo("unbounded semantic concept")

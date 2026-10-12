"""Checks on observed pixels, independent of synthetic path drawing programs."""
import pytest

from src.evaluation.structural_regression import evaluate_fixture, selective_metrics


def test_abstention_metrics_do_not_score_unknown_as_a_negative():
    result = selective_metrics([True, None, False], [True, False, False])
    assert result["correct"] == 2
    assert result["coverage"] == pytest.approx(2/3)
    assert result["selective_accuracy"] == 1
    assert result["overall_correct_fraction"] == pytest.approx(2/3)
    assert result["abstained"] == 1


def test_all_abstaining_accuracy_is_undefined():
    result = selective_metrics([None, None], [True, False])
    assert result["selective_accuracy"] is None
    assert result["coverage"] == 0
    assert result["overall_correct_fraction"] == 0


def test_metrics_reject_non_boolean_predictions_and_misalignment():
    with pytest.raises(TypeError):
        selective_metrics([.9], [True])
    with pytest.raises(ValueError):
        selective_metrics([True], [True, False])


@pytest.fixture(scope="module")
def supplied_problem_report():
    return evaluate_fixture()


def test_supplied_pixels_recover_expected_flow_without_shape_shortcuts(supplied_problem_report):
    result = supplied_problem_report["original"]
    assert result["coverage"] == 1
    assert result["overall_correct_fraction"] == 1
    assert all(panel["relation_correct"] for panel in result["panels"])
    assert all(panel["arrow_count"] == panel["expected_arrow_count"] for panel in result["panels"])


def test_open_and_closed_topology_is_recovered_from_pixels(supplied_problem_report):
    # Independent truth annotation of the visible diagrams. This set never
    # crosses the perception/search interface in evaluate_fixture.
    closed = {"A05.png", "B06.png"}
    for panel in supplied_problem_report["original"]["panels"]:
        loop = panel["file"] in closed
        assert panel["topology"] == ("closed" if loop else "open")
        assert panel["endpoint_count"] == (0 if loop else 2)
        assert panel["junction_count"] == 0
        assert panel["cycle_count"] == int(loop)


def test_each_unseen_pair_is_classified_by_its_independently_fitted_rule(supplied_problem_report):
    result = supplied_problem_report["heldout"]
    assert len(result["folds"]) == 6
    assert all(fold["coverage"] == 1 and fold["overall_correct_fraction"] == 1
               for fold in result["folds"])


def test_fixed_concept_survives_rigid_transforms_and_resizing(supplied_problem_report):
    for result in supplied_problem_report["transforms"]:
        assert result["coverage"] == 1, result["name"]
        assert result["overall_correct_fraction"] == 1, result["name"]


def test_example_order_and_group_names_are_not_perception_features(supplied_problem_report):
    assert supplied_problem_report["reordered"]["overall_correct_fraction"] == 1
    assert supplied_problem_report["labels_swapped"]["overall_correct_fraction"] == 1


def test_blank_or_contradictory_pixels_do_not_receive_a_separating_rule(supplied_problem_report):
    for control in supplied_problem_report["negative_controls"]:
        assert control["accepted_rule"] is None, control["name"]
        assert control["perfect_hypotheses"] == 0, control["name"]

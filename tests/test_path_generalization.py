"""Read rendered pixels, infer a rule, then freeze it for unseen scene seeds."""
import pytest
import numpy as np

from src.evaluation.path_generalization import evaluate_path_generalization
from src.active.path_generation import PathSceneGenerator
from src.dsl.nodes import Exists,Predicate
from src.scene.renderer import draw_scene
from src.vision.object_detection import extract_scene


@pytest.fixture(scope="module")
def report():
    return evaluate_path_generalization(training_seed=113,heldout_seeds=(7,31,91))


def test_all_supported_graph_demos_recover_from_page_pixels(report):
    for concept in report["concepts"]:
        assert concept["training"]["correct"]==12,concept["name"]
        assert concept["training"]["unknown"]==0,concept["name"]


def test_frozen_rules_predict_unseen_rendered_graphs_without_confident_errors(report):
    for concept in report["concepts"]:
        for measurement in (concept["aggregate_intended"],concept["aggregate_inferred"]):
            assert measurement["wrong"]==0,concept["name"]
            # Prevent trivial abstain-on-everything behavior while permitting
            # unresolved arrow evidence to remain honestly unavailable.
            assert measurement["coverage"]>=.90,concept["name"]
            assert measurement["total"]==36


def test_topology_and_curvature_have_full_heldout_coverage(report):
    for concept in report["concepts"]:
        if concept["name"] not in ("open_paths","straight_paths"): continue
        assert concept["aggregate_intended"]["correct"]==36,concept["name"]
        assert concept["aggregate_inferred"]["correct"]==36,concept["name"]


def test_ui_generated_positive_and_counterexample_read_back_from_pixels():
    rule=Exists(Predicate("arrow_relation","opposed"))
    generator=PathSceneGenerator(seed=97)
    for label in (True,False):
        symbolic=generator.sample(rule,label,2000)
        image=np.array(draw_scene(symbolic,280))
        assert rule.assess(symbolic) is label
        for adaptive in (False,True):
            perceived=extract_scene(image,adaptive=adaptive)
            assert rule.assess(perceived) is label

from dataclasses import asdict
import pytest
from src.dsl.nodes import Predicate, Count, Exists, And, Rule
from src.active import SceneGenerator


def test_ast_serialization():
    rule = And(Count(2), Exists(Predicate("shape", "circle")))
    assert Rule.from_dict(asdict(rule)) == rule
    with pytest.raises(ValueError): Rule.from_dict({"op": "execute_code"})


def test_presence_count_distribution_and_compositional_negatives():
    generator = SceneGenerator(17)
    presence = Exists(Predicate("shape", "circle"))
    examples = generator.balanced_examples(presence, 8)
    assert sorted(len(s.objects) for s, y in examples if y) == sorted(len(s.objects) for s, y in examples if not y)
    target = And(presence, Count(2))
    examples = generator.balanced_examples(target, 6)
    negative = [s for s, y in examples if not y]
    assert any(presence.evaluate(s) and not Count(2).evaluate(s) for s in negative)
    assert any(not presence.evaluate(s) and Count(2).evaluate(s) for s in negative)


def test_missing_human_data_does_not_produce_success():
    from src.evaluation.hand_drawn import evaluate_manifest
    report = evaluate_manifest("data/hand_drawn/manifest.example.json", "outputs/missing_data_check")
    assert report["real_problems_completed"] == 0
    assert report["errors"]


def test_rotation_ground_truth_retains_centers_and_spatial_labels():
    import numpy as np
    from src.evaluation.experiments import render_condition
    from src.scene import Scene, VisualObject, derive_relations
    scene = derive_relations(Scene([VisualObject(0, "triangle", True, center_x=.25, center_y=.3),
                                   VisualObject(1, "circle", False, center_x=.75, center_y=.7)]))
    image, rotated = render_condition(scene, "rotation", np.random.default_rng(42))
    assert image.size == (320, 320)
    assert [(o.center_x, o.center_y) for o in rotated.objects] == [(o.center_x, o.center_y) for o in scene.objects]
    assert all(b.angle == a.angle+18 for a, b in zip(scene.objects, rotated.objects))
    assert ("left_of", 0, 1) in rotated.relations

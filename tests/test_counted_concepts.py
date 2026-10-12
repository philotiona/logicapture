import numpy as np
import pytest

from src.active import AkinatorSession, SceneGenerator
from src.dsl.nodes import Predicate, Count, ForAll, And
from src.dsl.grammar import enumerate_rules
from src.dsl.predictions import prediction_matrix
from src.dsl.verbalizer import verbalize
from src.scene.renderer import draw_scene
from src.vision import extract_scene

FILLED_TRIANGLE = Predicate("and", children=(Predicate("shape", "triangle"), Predicate("filled")))


@pytest.mark.parametrize("seed", [1, 7, 42, 121])
@pytest.mark.parametrize("target", [Count(4, FILLED_TRIANGLE), Count(4, FILLED_TRIANGLE, True),
                                    And(Count(4), ForAll(FILLED_TRIANGLE))])
def test_four_filled_triangles_recovery(seed, target):
    session = AkinatorSession(seed=seed)
    assert target in session.rules
    target_predictions = session.predictions[session.rules.index(target)]
    assert target_predictions.any() and not target_predictions.all()
    while session.current is not None and len(session.history) < 30:
        session.answer(session.evaluate_query(target))
    assert session.current is None
    assert session.candidates == [target]
    assert any(answer.label for answer in session.history)


def test_counted_combo_generation_and_visible_objects():
    generator = SceneGenerator(55)
    target = Count(4, FILLED_TRIANGLE)
    for label in (True, False):
        scene = generator.sample(target, label, max_attempts=1)
        assert target.evaluate(scene) == label
        detected = extract_scene(np.array(draw_scene(scene)))
        assert target.evaluate(detected) == label
        assert len(detected.objects) == len(scene.objects)
    mixed = generator.predicate_scene(FILLED_TRIANGLE, 4, distractors=2)
    assert len(mixed.objects) == 6 and target.evaluate(mixed)
    assert not And(Count(4), ForAll(FILLED_TRIANGLE)).evaluate(mixed)
    assert target.evaluate(extract_scene(np.array(draw_scene(mixed))))
    assert verbalize(target) == "There are exactly 4 filled triangles"
    strict = And(Count(4), ForAll(FILLED_TRIANGLE))
    for label in (True, False):
        scene = generator.sample(strict, label, max_attempts=1)
        assert strict.evaluate(scene) == label
        assert strict.evaluate(extract_scene(np.array(draw_scene(scene)))) == label


def test_higher_counts_are_composed_for_other_shapes_too():
    outlined_circle = Predicate("and", children=(Predicate("shape", "circle"), Predicate("outline")))
    rule = Count(6, outlined_circle)
    assert rule in enumerate_rules()
    scene = SceneGenerator(5).sample(rule, True, max_attempts=1)
    assert rule.evaluate(scene)
    assert rule.evaluate(extract_scene(np.array(draw_scene(scene))))


def test_cached_predictions_match_executable_evaluator():
    scenes = SceneGenerator(20).pool(12)
    rules = enumerate_rules(depth=3)
    matrix = prediction_matrix(rules, scenes)
    probes = np.random.default_rng(3).choice(len(rules), 180, replace=False)
    for i in probes:
        assert np.array_equal(matrix[i], [rules[i].evaluate(s) for s in scenes])

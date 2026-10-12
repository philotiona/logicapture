import numpy as np
import pytest
from src.active import SceneGenerator, AkinatorSession, GenerationError
from src.dsl.nodes import Predicate, Exists, Count, And
from src.synthesis import search


def test_search_heldout_and_ambiguity():
    target = And(Exists(Predicate("shape", "circle")), Count(2))
    generator = SceneGenerator(121)
    examples = generator.balanced_examples(target, 12)
    result = search(examples)
    assert result.perfect_count >= 1
    assert result.ranked[0].correct == len(examples)
    probes = generator.pool(100)
    # Enough varied support to recover this composition semantically.
    assert sum(result.ranked[0].rule.evaluate(s) == target.evaluate(s) for s in probes) >= 95
    ambiguous = search(generator.balanced_examples(Count(2), 2))
    assert ambiguous.perfect_count > 1


def test_inconsistent_examples_return_failure_score():
    scene = SceneGenerator(3).random_scene(2)
    result = search([(scene, True), (scene, False)])
    assert result.perfect_count == 0
    assert result.ranked[0].accuracy == .5
    with pytest.raises(ValueError): search([(scene, True)])


def test_active_recovery_elimination_and_undo():
    target = Count(2)
    session = AkinatorSession(seed=33, pool_size=120)
    initial = len(session.survivors)
    index = session.current
    session.answer(session.evaluate_query(target))
    assert len(session.survivors) < initial
    session.undo()
    assert len(session.survivors) == initial and session.current == index
    for _ in range(25):
        if session.current is None: break
        previous = len(session.survivors)
        session.answer(session.evaluate_query(target))
        assert len(session.survivors) < previous
    assert target in session.candidates
    assert session.equivalence_count == 1


def test_answer_generation_and_timeout():
    generator = SceneGenerator(4)
    rule = Count(2)
    for label in (True, False): assert rule.evaluate(generator.sample(rule, label)) == label
    with pytest.raises(GenerationError): generator.sample(Count(9), True, max_attempts=3)


def test_active_query_balanced_partition():
    from src.active.query_selection import select_query
    matrix = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0]], bool)
    assert select_query(matrix, np.arange(4), np.arange(3)) == 1

import numpy as np
from src.active import AkinatorSession, SceneGenerator
from src.dsl.nodes import Count
from src.active.hypothesis_space import Answer


def test_positive_batch_keeps_individual_probes_and_undo():
    rules = [Count(1), Count(2), Count(3)]
    scenes = [SceneGenerator(i).random_scene(count=i) for i in (1, 2, 3)]
    session = AkinatorSession(rules=rules, scenes=scenes)
    session.current, session.query_indices = 0, (0, 1)
    session.answer(True)
    assert session.candidates == rules[:2]
    assert session.current in (0, 1)
    assert len(session.query_indices) == 1
    session.undo()
    assert session.query_indices == (0, 1)
    assert session.candidates == rules
    session.answer(False)
    assert session.candidates == [rules[2]]

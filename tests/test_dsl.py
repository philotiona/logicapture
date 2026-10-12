import pytest
from src.scene import VisualObject, Scene, derive_relations
from src.dsl.nodes import Predicate, Rule, Count, Exists, ForAll, Relation, And, Or, Not, Compare
from src.dsl.verbalizer import verbalize


@pytest.fixture
def scene():
    return derive_relations(Scene([VisualObject(0, "triangle", True, center_x=.2, center_y=.3, area=.02),
                                   VisualObject(1, "circle", False, center_x=.7, center_y=.65, area=.08)]))


@pytest.mark.parametrize("rule,expected", [
    (Exists(Predicate("shape", "circle")), True),
    (Exists(Predicate("shape", "quadrilateral")), False),
    (Count(2), True), (Count(2, Predicate("shape", "triangle")), False),
    (Count(1, Predicate("filled"), at_least=True), True),
    (ForAll(Predicate("filled")), False),
    (Relation("left_of", Predicate("shape", "triangle"), Predicate("shape", "circle")), True),
    (Relation("above", Predicate("shape", "circle"), Predicate("shape", "triangle")), False),
    (Relation("larger_than", Predicate("shape", "circle"), Predicate("shape", "triangle")), True),
    (Compare(Predicate("shape", "circle"), Predicate("shape", "triangle"), "eq"), True),
    (Not(Exists(Predicate("outline"))), False),
    (And(Count(2), Exists(Predicate("filled"))), True),
    (Or(Count(1), Count(2)), True),
    (Rule("implies", children=(Count(1), Count(3))), True),
])
def test_executable_rules(scene, rule, expected):
    assert rule.evaluate(scene) is expected
    assert rule.explain(scene)["matches"] is expected
    assert verbalize(rule)


def test_types_and_canonicalization():
    with pytest.raises(TypeError): Rule("and", children=(Predicate(), Predicate()))
    with pytest.raises(ValueError): Predicate("shape", "unicorn")
    with pytest.raises(ValueError): Count(-1)
    a, b = Count(2), Exists(Predicate("filled"))
    assert And(a, b) == And(b, a)
    assert Or(a, b) == Or(b, a)
    assert Not(Not(a)) == a


def test_empty_and_self_quantification():
    assert ForAll(Predicate("filled")).evaluate(Scene())
    assert Relation("inside", Predicate("filled"), Predicate("outline"), universal=True).evaluate(Scene())
    assert not Exists(Predicate()).evaluate(Scene())
    one = derive_relations(Scene([VisualObject(0, "circle", True)]))
    assert not Relation("same_shape").evaluate(one)


def test_universal_containment_and_witnesses():
    scene = derive_relations(Scene([VisualObject(0, "quadrilateral", False, area=.3), VisualObject(1, "circle", True, area=.015)]))
    rule = Relation("inside", Predicate("filled"), Predicate("outline"), universal=True)
    assert rule.evaluate(scene)
    assert rule.explain(scene)["witnesses"] == [(1, 0)]
    assert rule.explain(scene)["violating_ids"] == []

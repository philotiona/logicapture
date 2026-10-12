import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from src.active.path_generation import PathSceneGenerator, path_object, flip_arrow, transform_path
from src.dsl.grammar import enumerate_rules
from src.dsl.nodes import Predicate, Count, Exists, ForAll, And, Or, Not, Compare, Relation, Rule, uses_paths
from src.dsl.predictions import assessment_matrix, prediction_matrix
from src.dsl.verbalizer import verbalize
from src.scene import ArrowMark, PathEvidence, Scene, VisualObject,derive_relations
from src.scene.renderer import draw_scene


@pytest.mark.parametrize("family", ["line", "arc", "s", "wave", "loop", "irregular_loop"])
@pytest.mark.parametrize("signs,relation", [((1,1),"same"), ((1,-1),"opposed"), ((-1,-1),"same")])
def test_direction_is_along_curve_and_reversal_invariant(family, signs, relation):
    path = path_object(family,signs).path
    assert path.arrow_relation == relation
    assert path.reversed().arrow_relation == relation
    for angle in (37,90,178,270):
        for mirror in (False,True):
            moved = transform_path(path, angle=angle, mirror=mirror, scale=.8,offset=(.01,-.015))
            assert moved.arrow_relation == relation
            assert moved.topology == path.topology
            assert moved.geometry == path.geometry


def test_different_image_tangents_can_follow_same_path_direction():
    path = path_object("arc",(1,1)).path
    path = replace(path,arrows=(replace(path.arrows[0],arc_position=.2),replace(path.arrows[1],arc_position=.8)))
    a,b = [path.point_tangent(mark.arc_position)[1] for mark in path.arrows]
    assert np.dot(a,b) < 0  # Global image vectors alone would misclassify this.
    assert path.arrow_relation == "same"


@pytest.mark.parametrize("marks", [(), (ArrowMark(.5,1,1),),
                                    (ArrowMark(.3,1,1),ArrowMark(.7,0,.4)),
                                    (ArrowMark(.3,1,.4),ArrowMark(.7,-1,1))])
def test_missing_direction_is_unknown_even_when_negated(marks):
    obj = replace(path_object(),path=PathEvidence(((.1,.3),(.5,.8),(.9,.3)),"open",marks))
    scene = Scene([obj])
    for relation in ("same","opposed"):
        rule = Exists(Predicate("arrow_relation",relation))
        assert rule.assess(scene) is None
        assert Not(rule).assess(scene) is None
        assert not rule.evaluate(scene)
        assert rule.explain(scene)["unknown_ids"] == [0]
    assert obj.shape == "unknown"
    assert Exists(Predicate("shape","triangle")).assess(scene) is None
    assert Exists(Predicate("outline")).assess(scene) is None


def test_count_uses_intervals_with_unknown_evidence():
    known = path_object("s",(1,-1))
    unknown = replace(path_object("line",()),id=1)
    scene = Scene([known,unknown])
    opposed = Predicate("arrow_relation","opposed")
    assert Count(0,opposed).assess(scene) is False
    assert Count(1,opposed).assess(scene) is None
    assert Count(2,opposed).assess(scene) is None
    assert Count(3,opposed).assess(scene) is False
    assert Count(1,opposed,at_least=True).assess(scene) is True
    assert Count(2,opposed,at_least=True).assess(scene) is None
    assert Count(3,opposed,at_least=True).assess(scene) is False
    assert ForAll(opposed).assess(scene) is None
    assert Compare(opposed,opposed,"eq").assess(scene) is True


def test_path_contract_round_trip_and_old_scenes():
    obj = path_object("loop",(1,-1))
    arrow = replace(obj.path.arrows[0],tip=(.1,.2),tail=(.2,.3),probabilities={"forward":.8,"backward":.1,"unknown":.1})
    obj = replace(obj,path=replace(obj.path,arrows=(arrow,obj.path.arrows[1]),
                                 topology_probabilities={"closed":.98,"open":.02}))
    restored = Scene.from_dict(json.loads(json.dumps(Scene([obj]).to_dict())))
    assert restored.objects == [obj]
    legacy = Scene.from_dict({"objects":[{"id":0,"shape":"circle","filled":True}]})
    assert legacy.objects[0].kind == "region"
    assert legacy.objects[0].path is None
    with pytest.raises(ValueError): replace(obj,shape="triangle")
    with pytest.raises(ValueError): ArrowMark(.4,2,1)


def test_cached_evaluators_agree_including_unknown_and_composition():
    same,opposed = [Exists(Predicate("arrow_relation",value)) for value in ("same","opposed")]
    scenes = [Scene([path_object("s",(1,1))]),Scene([path_object("s",(1,-1))]),
              Scene([path_object("s",())]),Scene([VisualObject(0,"triangle",False)]),Scene()]
    rules = [same,opposed,Not(opposed),And(same,opposed),Or(same,opposed),Count(1,Predicate("arrow_relation","same")),
             Compare(Predicate("arrow_relation","same"),Predicate("arrow_relation","opposed")),
             ForAll(Predicate("arrow_relation","opposed")),Relation("inside"),
             Rule("implies",children=(same,opposed))]
    observed = assessment_matrix(rules,scenes)
    expected = np.array([[-1 if (v:=rule.assess(scene)) is None else int(v) for scene in scenes] for rule in rules])
    assert np.array_equal(observed,expected)
    assert np.array_equal(prediction_matrix(rules,scenes),expected==1)


def test_graph_grammar_is_opt_in_compositional_and_serializable():
    before,after = enumerate_rules(depth=1),enumerate_rules(depth=1,paths=True)
    assert not any(uses_paths(rule) for rule in before)
    assert set(before).issubset(after)
    expected = ForAll(Predicate("arrow_relation","opposed"))
    assert expected in after
    assert Rule.from_dict(asdict(expected)) == expected
    assert "opposite ways" in verbalize(expected)
    compound = ForAll(Predicate("and",children=(Predicate("topology","closed"),Predicate("arrow_relation","same"))))
    assert compound in after
    assert "closed path" in verbalize(compound)


@pytest.mark.parametrize("rule", [Exists(Predicate("arrow_relation","opposed")),
                                  ForAll(Predicate("arrow_relation","same")),
                                  Count(2,Predicate("arrow_relation","opposed")),
                                  ForAll(Predicate("topology","open")),
                                  Exists(Predicate("geometry","straight")),
                                  ForAll(Predicate("has_arrows")),
                                  And(Count(1),ForAll(Predicate("arrow_relation","opposed")))])
def test_checked_generation_and_counterexamples(rule):
    examples = PathSceneGenerator(seed=14).balanced_examples(rule)
    assert len(examples) == 12
    assert all(rule.assess(scene) is label for scene,label in examples)
    assert all(draw_scene(scene).size == (320,320) for scene,_ in examples)
    if rule.op in ("exists","forall") and rule.predicates[0].op == "arrow_relation":
        for (a,_),(b,_) in zip(examples[:6],examples[6:]):
            assert a.objects[0].path.points == b.objects[0].path.points
            assert len(a.objects[0].path.arrows) == len(b.objects[0].path.arrows)


def test_unknown_region_does_not_become_confident_not_triangle():
    scene = Scene([VisualObject(0,"unknown",False,kind="unknown")])
    assert Not(Exists(Predicate("shape","triangle"))).assess(scene) is None
    assert Exists(Predicate("kind","path")).assess(scene) is None


def test_topology_survives_ambiguous_branched_traversal():
    obj = path_object("branch",())
    obj = replace(obj,path=replace(obj.path,confidence=.35,topology_confidence=.95))
    scene = Scene([obj])
    assert ForAll(Predicate("topology","branched")).assess(scene) is True
    assert ForAll(Predicate("endpoint_count","3")).assess(scene) is True
    assert ForAll(Predicate("junction_count","1")).assess(scene) is True
    assert ForAll(Predicate("cycle_count","0")).assess(scene) is True
    assert Exists(Predicate("arrow_relation","opposed")).assess(scene) is None
    assert Scene.from_dict(Scene([obj]).to_dict()).objects==[obj]


@pytest.mark.parametrize("predicate",[Predicate("topology","open"),Predicate("topology","closed"),
                                     Predicate("geometry","curved"),Predicate("has_arrows")])
def test_graph_examples_keep_domain_in_both_groups(predicate):
    for scene,_ in PathSceneGenerator(seed=11).balanced_examples(ForAll(predicate)):
        assert scene.objects
        assert all(obj.kind=="path" for obj in scene.objects)


def test_unusual_closed_region_geometry_is_independent_of_shape_class():
    outer=VisualObject(0,"unknown",False,area=.3,contour=((.1,.1),(.8,.1),(.8,.3),(.4,.3),(.4,.8),(.1,.8)))
    inner=VisualObject(1,"unknown",True,area=.01,center_x=.2,center_y=.4,
                       contour=((.16,.36),(.24,.36),(.24,.44),(.16,.44)))
    scene=derive_relations(Scene([outer,inner]))
    assert ("inside",1,0) in scene.relations
    rule=Relation("inside")
    assert rule.assess(scene) is True
    assert rule.explain(scene)["witnesses"]==[(1,0)]
    assert Relation("same_shape").assess(scene) is None


def test_generated_arrowheads_have_separate_attachment_regions():
    for family in ("line","arc","s","wave","loop","irregular_loop"):
        for signs in ((1,-1),(1,1),(1,-1,1)):
            positions=[a.arc_position for a in path_object(family,signs).path.arrows]
            assert all(b-a>=.14 for a,b in zip(positions,positions[1:]))


def test_marker_presence_and_count_do_not_require_known_direction():
    marks=(ArrowMark(.3,0,.4,presence_confidence=.9),ArrowMark(.7,0,.5,presence_confidence=.95))
    obj=replace(path_object(),path=PathEvidence(((.1,.3),(.9,.3)),"open",marks))
    assert Predicate("has_arrows").assess(obj) is True
    assert Predicate("arrow_count","2").assess(obj) is True
    assert Predicate("arrow_relation","same").assess(obj) is None
    assert obj.path.arrow_relation=="unknown"
    assert Scene.from_dict(Scene([obj]).to_dict()).objects==[obj]
    unresolved=replace(obj,path=replace(obj.path,topology="unknown",confidence=.4))
    assert Predicate("has_arrows").assess(unresolved) is True
    assert Predicate("arrow_count","2").assess(unresolved) is None
    assert Predicate("arrow_relation","same").assess(unresolved) is None


def test_boolean_cache_does_not_negate_missing_closed_boundary_evidence():
    obj=VisualObject(0,"quadrilateral",False,path=PathEvidence(((.1,.1),(.9,.1),(.9,.9),(.1,.9),(.1,.1)),"closed"))
    scene=Scene([obj])
    rule=Not(Exists(Predicate("arrow_relation","same")))
    assert rule.assess(scene) is None
    assert prediction_matrix([rule],[scene])[0,0] == rule.evaluate(scene)

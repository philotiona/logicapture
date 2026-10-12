import numpy as np
import pytest
from src.scene import VisualObject, Scene, GeometryTolerance, derive_relations
from src.scene.renderer import draw_scene


def test_scene_roundtrip_and_renderer():
    scene = derive_relations(Scene([VisualObject(4, "triangle", False, area=.04)]))
    assert Scene.from_dict(scene.to_dict()).to_dict() == scene.to_dict()
    assert np.array_equal(np.asarray(draw_scene(scene)), np.asarray(draw_scene(scene)))
    with pytest.raises(ValueError): Scene([scene.objects[0], scene.objects[0]])
    with pytest.raises(ValueError): VisualObject(0, "circle", True, center_x=4)


def test_relation_tolerances_and_alignment():
    scene = Scene([VisualObject(i, "circle", True, center_x=.25+i*.2, center_y=.5, area=.01) for i in range(3)])
    derive_relations(scene)
    assert ("left_of", 0, 1) in scene.relations
    assert ("near", 0, 1) in scene.relations
    assert ("same_shape", 0, 2) in scene.relations
    assert ("aligned", 0, 1, 2) in scene.relations
    derive_relations(scene, GeometryTolerance(position_margin=.25))
    assert ("left_of", 0, 1) not in scene.relations


def test_containment_uses_boundary_not_center():
    scene = derive_relations(Scene([VisualObject(0, "circle", False, area=.03), VisualObject(1, "triangle", True, area=.028)]))
    assert ("inside", 1, 0) not in scene.relations

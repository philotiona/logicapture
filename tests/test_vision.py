import numpy as np
import pytest
from src.scene import Scene, VisualObject, derive_relations
from src.scene.renderer import draw_scene, draw_page
from src.vision import extract_scene, detect_panels, rectify_page


@pytest.mark.parametrize("shape", ["circle", "triangle", "quadrilateral"])
@pytest.mark.parametrize("filled", [True, False])
def test_shape_and_fill_roundtrip(shape, filled):
    scene = Scene([VisualObject(0, shape, filled, area=.065, angle=16)])
    detected = extract_scene(np.array(draw_scene(scene)))
    assert len(detected.objects) == 1
    assert detected.objects[0].shape == shape
    assert detected.objects[0].filled == filled
    assert abs(detected.objects[0].center_x-.5) < .015


def test_nested_contours_keep_inner_object():
    scene = Scene([VisualObject(0, "quadrilateral", False, area=.30), VisualObject(1, "circle", True, area=.015)])
    detected = extract_scene(np.array(draw_scene(scene)))
    assert len(detected.objects) == 2
    assert sorted(o.filled for o in detected.objects) == [False, True]
    assert any(r[0] == "inside" for r in detected.relations)


def test_panel_labels_and_page_pipeline():
    circle = Scene([VisualObject(0, "circle", True, area=.04)])
    triangle = Scene([VisualObject(0, "triangle", False, area=.04)])
    page = np.array(draw_page(([circle]*6, [triangle]*6)))
    panels = detect_panels(page)
    assert len(panels) == 12
    for panel in panels:
        objects = extract_scene(panel.image).objects
        assert len(objects) == 1
        assert objects[0].shape == ("circle" if panel.label else "triangle")
    with pytest.raises(ValueError): detect_panels(np.full((400, 600, 3), 255, np.uint8))


def test_rectification_manual_corners():
    image = np.full((300, 400, 3), 255, np.uint8)
    corrected, applied = rectify_page(image, [[20, 20], [380, 30], [370, 280], [25, 275]])
    assert applied and corrected.shape[0] > 240 and corrected.shape[1] > 340


def test_color_and_relation_roundtrip():
    scene = derive_relations(Scene([VisualObject(0, "triangle", True, "red", .25, .3, .03),
                                   VisualObject(1, "circle", False, "blue", .75, .7, .04)]))
    detected = extract_scene(np.array(draw_scene(scene)))
    assert {o.color for o in detected.objects} == {"red", "blue"}
    triangle = next(o for o in detected.objects if o.shape == "triangle")
    circle = next(o for o in detected.objects if o.shape == "circle")
    assert ("left_of", triangle.id, circle.id) in detected.relations
    assert ("above", triangle.id, circle.id) in detected.relations

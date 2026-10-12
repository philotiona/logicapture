"""Input-support and OOD regressions for the default evidence-first pipeline."""
import cv2
import numpy as np

from src.scene import Scene, VisualObject, derive_relations
from src.scene.renderer import draw_scene
from src.vision import extract_scene
from src.vision.adaptive_perception import remove_ruling


class ContradictoryModel:
    def __init__(self): self.calls = 0

    def __call__(self, images, training=False):
        self.calls += 1
        return {"shape": np.tile([.01, .97, .01, .01], (len(images), 1)),
                "fill": np.tile([.01, .99], (len(images), 1))}


def test_open_curve_is_not_a_cnn_triangle():
    image = np.full((200, 200, 3), 255, np.uint8)
    points = np.array([[145, 40], [60, 60], [40, 95], [100, 110], [140, 145], [100, 165]], np.int32)
    cv2.polylines(image, [points], False, (0, 0, 0), 3)
    model = ContradictoryModel()
    scene = extract_scene(image, attribute_model=model, uncertain_only=False)
    assert len(scene.objects) == 1
    assert scene.objects[0].kind == "path" and scene.objects[0].shape == "unknown"
    assert model.calls == 0
    assert scene.objects[0].path.arrow_relation == "unknown"


def test_structural_evidence_guards_contradictory_closed_set_model():
    image = np.array(draw_scene(Scene([VisualObject(0, "circle", False, area=.08)])))
    model = ContradictoryModel()
    obj = extract_scene(image, attribute_model=model, uncertain_only=False).objects[0]
    assert obj.shape == "circle" and not obj.filled
    assert obj.shape_probabilities["triangle"] == .97
    assert obj.attribute_source == "neural_rejected"


def test_thin_straight_stroke_survives_zero_enclosed_area():
    image = np.full((200, 200, 3), 255, np.uint8)
    cv2.line(image, (30, 100), (170, 100), (0, 0, 0), 1)
    scene = extract_scene(image)
    assert len(scene.objects) == 1 and scene.objects[0].kind == "path"
    assert scene.objects[0].path.topology == "open"


def test_free_form_filled_region_retains_contour_without_forced_shape():
    image = np.full((200, 200, 3), 255, np.uint8)
    points = np.array([[40, 40], [160, 40], [160, 160], [130, 160],
                       [130, 70], [70, 70], [70, 160], [40, 160]], np.int32)
    cv2.fillPoly(image, [points], (0, 0, 0))
    model = ContradictoryModel()
    scene = extract_scene(image, attribute_model=model, uncertain_only=False)
    assert len(scene.objects) == 1
    assert scene.objects[0].kind == "region" and scene.objects[0].shape == "unknown"
    assert scene.objects[0].filled and scene.objects[0].contour
    assert model.calls == 0


def test_concave_notch_is_not_filled_by_a_convex_hull_relation():
    u = ((.2, .2), (.8, .2), (.8, .8), (.65, .8), (.65, .35), (.35, .35), (.35, .8), (.2, .8))
    small = ((.44, .55), (.56, .55), (.56, .67), (.44, .67))
    scene = derive_relations(Scene([VisualObject(0, "unknown", True, area=.20, contour=u),
                                   VisualObject(1, "quadrilateral", True, area=.0144, contour=small)]))
    assert not any(relation[0] in ("inside", "overlapping", "touching") for relation in scene.relations)


def test_ruling_removal_preserves_an_interior_l_stroke():
    mask = np.zeros((200, 200), np.uint8)
    cv2.line(mask, (20, 180), (180, 180), 255, 1)
    cv2.line(mask, (180, 20), (180, 180), 255, 1)
    cv2.line(mask, (50, 60), (50, 120), 255, 3)
    cv2.line(mask, (50, 120), (115, 120), 255, 3)
    clean, count = remove_ruling(mask)
    assert count == 1 and not clean[180, 180]
    assert clean[120, 80] == 255


def test_vertices_inside_do_not_imply_an_edge_bridging_a_notch_is_inside():
    u = ((.1, .1), (.3, .1), (.3, .7), (.7, .7), (.7, .1), (.9, .1), (.9, .9), (.1, .9))
    bar = ((.2, .2), (.8, .2), (.8, .3), (.2, .3))
    scene = derive_relations(Scene([VisualObject(0, "unknown", True, area=.4, contour=u),
                                   VisualObject(1, "unknown", True, area=.06, contour=bar)]))
    assert ("inside", 1, 0) not in scene.relations
    assert ("overlapping", 0, 1) in scene.relations


def test_default_search_cannot_use_unknown_shape_as_a_confident_negative():
    from src.synthesis import search
    examples = [(Scene([VisualObject(0, "triangle", False)]), True),
                (Scene([VisualObject(0, "unknown", False)]), False)]
    result = search(examples)
    assert result.perfect_count == 0
    for ranked in result.ranked:
        values = [ranked.rule.assess(scene) for scene, _ in examples]
        assert ranked.unknown == sum(value is None for value in values)
        assert ranked.correct == sum(value is label for value, (_, label) in zip(values, examples))


def test_free_form_outline_retains_a_region_view_for_containment():
    from src.dsl.nodes import Predicate, Relation
    image = np.full((300, 300, 3), 255, np.uint8)
    boundary = np.array([[45, 45], [215, 45], [245, 105], [215, 240],
                         [150, 220], [55, 245], [75, 130]], np.int32)
    cv2.polylines(image, [boundary], True, (0, 0, 0), 3)
    cv2.circle(image, (145, 125), 14, (0, 0, 0), -1)
    scene = extract_scene(image)
    assert len(scene.objects) == 2
    outer = max(scene.objects, key=lambda obj: obj.area)
    assert outer.kind == "region" and outer.shape == "unknown" and outer.path is not None
    assert Relation("inside", Predicate("filled"), Predicate("outline")).assess(scene) is True

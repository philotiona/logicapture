import numpy as np
import pytest
from src.scene import Scene, VisualObject
from src.scene.renderer import draw_scene
from src.vision import extract_scene
from src.vision.attribute_cnn import predict_attributes, normalize_region
from src.dsl.nodes import Exists, Predicate


class StubAttributeModel:
    def __init__(self): self.batches = []

    def __call__(self, batch, training=False):
        self.batches.append(batch)
        return {"shape": np.tile([.05, .85, .05, .05], (len(batch), 1)),
                "fill": np.tile([.2, .8], (len(batch), 1))}


def test_probability_contract_and_hard_argmax():
    original = Scene([VisualObject(0, "circle", False, area=.08)])
    image = np.array(draw_scene(original))
    model = StubAttributeModel()
    # The deliberately contradictory stub exercises the retained argmax
    # baseline. The default structural pipeline guards incompatible labels.
    neural = extract_scene(image, attribute_model=model, uncertain_only=False, representation="geometric")
    obj = neural.objects[0]
    assert obj.shape == "triangle" and obj.filled is True
    assert obj.shape_probabilities["triangle"] == .85
    assert obj.fill_probabilities["filled"] == .8
    assert obj.attribute_source == "neural_argmax"
    assert Exists(Predicate("shape", "triangle")).evaluate(neural)
    assert Scene.from_dict(neural.to_dict()).to_dict() == neural.to_dict()
    baseline = extract_scene(image, representation="geometric")
    assert baseline.objects[0].shape == "circle"
    assert baseline.objects[0].shape_probabilities is None
    assert baseline.objects[0].attribute_source == "opencv"


def test_uncertainty_gate_can_preserve_certain_opencv_attributes():
    image = np.array(draw_scene(Scene([VisualObject(0, "triangle", True, area=.06)])))
    model = StubAttributeModel()
    scene = extract_scene(image, attribute_model=model, uncertainty_threshold=.8)
    assert scene.objects[0].attribute_source == "opencv"
    assert model.batches == []


def test_bad_distributions_and_output_shape():
    with pytest.raises(ValueError): VisualObject(0, "circle", True, shape_probabilities={"circle": .4})
    with pytest.raises(ValueError): VisualObject(0, "circle", True, fill_probabilities={"filled": 1.1, "outline": -.1})
    region = np.zeros((80, 30), np.uint8)
    assert normalize_region(region).shape == (48, 48, 1)
    probabilities = predict_attributes(StubAttributeModel(), [region])
    assert abs(sum(probabilities[0][0].values())-1) < 1e-6


def test_nested_regions_do_not_include_another_objects_ink():
    image = np.array(draw_scene(Scene([VisualObject(0, "quadrilateral", False, area=.3),
                                     VisualObject(1, "circle", True, area=.015)])))
    model = StubAttributeModel()
    scene = extract_scene(image, attribute_model=model, uncertain_only=False)
    assert len(scene.objects) == 2
    regions = model.batches[0]
    # Segmentation preserves the white center of the outer outline, even when
    # the photographed scene has a filled object nested inside it.
    assert any(region[24, 24, 0] > .9 for region in regions)

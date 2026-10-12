import numpy as np
import pytest
from pathlib import Path
from src.scene import Scene, VisualObject
from src.scene.renderer import draw_scene
from src.vision import extract_scene


def test_saved_object_cnn_integrates_with_real_segmentation():
    pytest.importorskip("tensorflow")
    if not Path("data/models/attributes.keras").exists(): pytest.skip("Object model not trained yet")
    from src.vision.attribute_cnn import load_attribute_model
    model = load_attribute_model()
    image = np.array(draw_scene(Scene([VisualObject(0, "circle", True, area=.055, angle=17)])))
    scene = extract_scene(image, attribute_model=model, uncertain_only=False)
    assert len(scene.objects) == 1
    obj = scene.objects[0]
    assert obj.shape == "circle" and obj.filled
    assert obj.attribute_source == "neural_argmax"
    assert abs(sum(obj.shape_probabilities.values())-1) < 1e-5
    assert abs(sum(obj.fill_probabilities.values())-1) < 1e-5

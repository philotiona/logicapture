from dataclasses import replace

import numpy as np
import pytest

from src.rgr.inference import (CLASSES, MODEL_PATH, ROOT, Settings, evaluate_images,
                               metrics, parameter_sweeps, predict_image,
                               prepare_model_input, transform_image)


def shape_image():
    image = np.full((80, 120, 3), 255, np.uint8)
    image[20:60, 35:85] = 0
    image[20:30, 35:50] = 100
    return image


class Model:
    def __init__(self, probabilities=(.1, .2, .65, .05)):
        self.probabilities = probabilities
        self.inputs = []

    def __call__(self, batch, training=False):
        assert training is False
        self.inputs.append(batch.copy())
        return {"shape": np.asarray([self.probabilities]), "fill": np.asarray([[.4, .6]])}


def test_exact_shared_pipeline_and_confidence_threshold():
    image, model = shape_image(), Model()
    result = predict_image(image, model)
    assert result["prediction"] == "quadrilateral" and result["accepted"]
    assert result["confidence"] == .65
    assert result["probabilities"] == dict(zip(CLASSES, (.1, .2, .65, .05)))
    assert result["elapsed_ms"] >= 0
    assert result["model_input_size"] == [48, 48] and result["processed_size"] == 224
    np.testing.assert_array_equal(model.inputs[0][0], prepare_model_input(image))
    assert model.inputs[0].shape == (1, 48, 48, 1)
    rejected = predict_image(image, model, Settings(threshold=.7))
    assert not rejected["accepted"] and rejected["prediction"] == result["prediction"]
    assert rejected["confidence"] == result["confidence"]
    assert ROOT.is_absolute() and MODEL_PATH == ROOT/"data/models/attributes.keras"


def test_brightness_contrast_and_resolution_reach_the_model_unchanged():
    image, model = shape_image(), Model()
    original = transform_image(image)
    assert original.shape == (224, 224, 3)
    assert original.dtype == np.uint8
    # The non-square source is centered with white padding, without stretching.
    assert np.all(original[:30] == 255)
    for settings in (Settings(brightness=-50), Settings(contrast=.5), Settings(size=128),
                     Settings(rotation=20), Settings(flip=True)):
        transformed = transform_image(image, settings)
        predict_image(image, model, settings)
        np.testing.assert_array_equal(model.inputs[-1][0], prepare_model_input(image, settings))
        if settings.size == 224:
            assert not np.array_equal(transformed, original)
    assert np.mean(prepare_model_input(image, Settings(brightness=-50))) < np.mean(prepare_model_input(image))
    assert prepare_model_input(image, Settings(contrast=.5)).min() > prepare_model_input(image).min()


@pytest.mark.parametrize("image", [np.full((32, 32, 3), 255, np.uint8),
                                   np.zeros((32, 32), np.uint8),
                                   np.full((20, 40, 4), 0, np.uint8)])
def test_blank_inputs_are_rejected_without_closed_set_model_guess(image):
    model = Model()
    result = predict_image(image, model)
    assert result["prediction"] is None and result["accepted"] is False
    assert result["reason"] == "blank_or_uniform_image"
    assert result["probabilities"] == {} and not model.inputs


def test_brightness_saturation_is_rejected():
    model = Model()
    for brightness in (-100, 100):
        assert not predict_image(shape_image(), model, Settings(brightness=brightness))["accepted"]
    assert not model.inputs


@pytest.mark.parametrize("probabilities", [(np.nan, .2, .7, .1), (.1, -.1, .7, .3),
                                          (.1, .2, .3, .1), (.1, .2, .7), (0, 0, 2, -1)])
def test_invalid_probability_heads_fail_explicitly(probabilities):
    with pytest.raises(ValueError, match="shape head"):
        predict_image(shape_image(), Model(probabilities))


def test_evaluation_reports_rejections_separately_and_does_not_inflate_accuracy():
    image, model = shape_image(), Model()
    rows = evaluate_images([image, image, np.full_like(image, 255)],
                           ["quadrilateral", "circle", "circle"], model)
    report = metrics(rows)
    assert report["total"] == 3 and report["correct"] == 1
    assert report["errors"] == 1 and report["rejected"] == 1
    assert report["accuracy"] == 1/3 and report["coverage"] == 2/3
    rejected = evaluate_images([image], ["quadrilateral"], model, Settings(threshold=.7))
    assert metrics(rejected)["accuracy"] == 0 and metrics(rejected)["raw_accuracy"] == 1
    assert metrics([])["total"] == 0 and metrics([])["accuracy"] == 0
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_images([image], [], model)
    with pytest.raises(ValueError, match="Ground-truth"):
        evaluate_images([image], ["unlisted"], model)


def test_parameter_sweeps_use_real_outputs_and_all_requested_values():
    model = Model()
    sweeps = parameter_sweeps(shape_image(), model)
    assert set(sweeps) == {"threshold", "brightness", "contrast", "size", "augmentation"}
    assert [row["value"] for row in sweeps["threshold"]] == [.1, .3, .5, .7, .9]
    assert [row["accepted"] for row in sweeps["threshold"]] == [True, True, True, False, False]
    assert [row["value"] for row in sweeps["brightness"]] == [-100, -50, 0, 50, 100]
    assert [row["value"] for row in sweeps["contrast"]] == [.5, 1., 2.]
    assert [row["processed_size"] for row in sweeps["size"]] == [128, 224, 320, 416, 640]
    assert [row["value"] for row in sweeps["augmentation"]] == ["original", "rotate20", "hflip"]
    assert all(row["confidence"] == .65 for row in sweeps["threshold"])
    assert len(model.inputs) == 19  # Saturated brightness endpoints do not invoke the CNN.


@pytest.mark.parametrize("changes", [{"threshold": np.nan}, {"threshold": 1.1}, {"brightness": 101},
                                    {"contrast": .2}, {"size": 48}, {"rotation": np.inf}])
def test_invalid_settings_fail_early(changes):
    with pytest.raises(ValueError):
        replace(Settings(), **changes)

"""Canonical pixels, resilient local caches, and shared encoder integration.

All semantic inference here is a tiny fake tensor/model protocol.  These tests
never load weights, make network requests, or examine held-out dataset labels.
"""
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.reasoning.general import GeneralVisualSolver, validate_episode_images
from src.reasoning.features import FeatureBatch
from src.reasoning.concepts import support_sheet
from src.vision.pretrained import SiglipEncoder, image_digest, rgb_image
from src.vision.structural_features import StructuralEncoder, StructuralFeatureEncoder


def _solid(value, side=16):
    return np.full((side, side, 3), value, np.uint8)


def test_rgba_uses_visible_white_background_for_features_and_cache_identity():
    rgba = np.zeros((16, 16, 4), np.uint8)
    rgba[:, 8:, 0] = 255
    rgba[:, 8:, 3] = 128
    original = rgba.copy()
    visible = np.full((16, 16, 3), 255, np.uint8)
    visible[:, 8:] = (255, 127, 127)
    np.testing.assert_array_equal(np.asarray(rgb_image(rgba)), visible)
    np.testing.assert_array_equal(np.asarray(rgb_image(Image.fromarray(rgba))), visible)
    np.testing.assert_array_equal(rgba, original)
    assert image_digest(rgba) == image_digest(visible)
    encoded = StructuralEncoder(side=64).encode([rgba, visible])
    np.testing.assert_array_equal(encoded.values[0], encoded.values[1])
    np.testing.assert_array_equal(encoded.valid[0], encoded.valid[1])


def test_pil_transparency_and_exif_are_canonicalized_before_pixel_hashing():
    indexed = Image.new("P", (16, 16), color=0)
    indexed.putpalette([0, 0, 0]+[255, 0, 0]*255)
    indexed.info["transparency"] = 0
    assert image_digest(indexed) == image_digest(_solid(255))
    raw = np.zeros((12, 20, 3), np.uint8)
    raw[:4, 2:8] = (80, 160, 240)
    tagged = Image.fromarray(raw)
    tagged.getexif()[274] = 6  # Rotate clockwise for display.
    expected = np.rot90(raw, 3).copy()
    np.testing.assert_array_equal(np.asarray(rgb_image(tagged)), expected)
    assert image_digest(tagged) == image_digest(expected)


def test_language_support_sheet_uses_the_same_visible_alpha_pixels():
    transparent = np.zeros((16, 16, 4), np.uint8)
    sheet = np.asarray(support_sheet([transparent], [_solid(220)], tile=64))
    # Centers of the two image tiles, away from labels and frames.
    np.testing.assert_array_equal(sheet[66:74, 36:44], np.full((8, 8, 3), 255, np.uint8))
    np.testing.assert_array_equal(sheet[170:178, 36:44], np.full((8, 8, 3), 220, np.uint8))


@pytest.mark.parametrize("alias", ["gray", "gray_channel", "rgba", "pil_gray"])
def test_unseen_support_duplicates_are_detected_across_channel_formats(alias):
    positive = [_solid(20), _solid(30)]
    negative = [_solid(220), _solid(230)]
    gray = positive[0][:, :, 0]
    variants = {"gray": gray, "gray_channel": gray[:, :, None],
                "rgba": np.dstack((positive[0], np.full(gray.shape, 255, np.uint8))),
                "pil_gray": Image.fromarray(gray)}
    with pytest.raises(ValueError, match="duplicates"):
        validate_episode_images(positive, negative, [variants[alias]])


def test_episode_inputs_are_shared_canonical_rgb_and_invalid_channels_fail_early():
    transparent = np.zeros((16, 16, 4), np.uint8)
    canonical = validate_episode_images([_solid(20), _solid(30)],
                                        [_solid(210), _solid(220)], [transparent])
    assert all(image.dtype == np.uint8 and image.shape == (16, 16, 3) for image in canonical)
    np.testing.assert_array_equal(canonical[-1], _solid(255))
    with pytest.raises(ValueError, match="channels"):
        validate_episode_images([_solid(20), _solid(30)], [_solid(210), _solid(220)],
                                [np.zeros((16, 16, 2), np.uint8)])


class _Tensor:
    def __init__(self, values): self.values = np.asarray(values, np.float32)
    def to(self, *args, **kwargs): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.values


class _Processor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": _Tensor([np.asarray(image).mean()/255 for image in images])}


class _VisionModel:
    def __init__(self): self.calls = 0
    def parameters(self): return iter((SimpleNamespace(dtype="fake-float32"),))
    def __call__(self, *, pixel_values, return_dict):
        assert return_dict
        self.calls += 1
        brightness = pixel_values.values
        vector = np.repeat(brightness[:, None], 768, axis=1)
        patches = np.repeat(vector[:, None, :], 196, axis=1)
        return SimpleNamespace(pooler_output=_Tensor(vector), last_hidden_state=_Tensor(patches))


@pytest.mark.parametrize("corruption", ["broken_zip", "shape", "nonfinite", "revision"])
def test_bad_feature_cache_is_recomputed_atomically_without_model_weights(tmp_path, monkeypatch, corruption):
    encoder = SiglipEncoder(cache_dir=tmp_path/"no-models", feature_cache=tmp_path/"features")
    picture = _solid(85)
    path = encoder._cache_path(image_digest(picture))
    path.parent.mkdir(parents=True)
    if corruption == "broken_zip":
        path.write_bytes(b"PK\x03\x04incomplete archive")
    else:
        vector = np.ones(5 if corruption == "shape" else 768, np.float32)
        if corruption == "nonfinite": vector[0] = np.nan
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{"global": vector, "patches": np.ones((196, 768), np.float32),
                                          "encoder_id": "old-revision" if corruption == "revision" else encoder.encoder_id})
    model, loads = _VisionModel(), []
    def fake_load():
        loads.append(True)
        encoder._model, encoder._processor, encoder.device = model, _Processor(), "cpu"
    monkeypatch.setattr(encoder, "_load", fake_load)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    repaired = encoder.encode([picture], ["first-id"])
    assert loads == [True] and model.calls == 1
    np.testing.assert_allclose(repaired.global_features, 85/255)
    assert not tuple(path.parent.glob("*.tmp"))
    with np.load(path, allow_pickle=False) as saved:
        assert str(saved["encoder_id"]) == encoder.encoder_id
        assert saved["global"].shape == (768,)
        assert saved["patches"].shape == (196, 768)
        assert np.isfinite(saved["global"]).all()
    # A different image identifier must reuse the repaired pixel cache.  Any
    # second inference or attempt to load weights is a regression.
    def forbidden_load(): raise AssertionError("A valid cache attempted to load model weights")
    monkeypatch.setattr(encoder, "_load", forbidden_load)
    cached = encoder.encode([picture], ["different-id"])
    assert model.calls == 1
    np.testing.assert_array_equal(cached.global_features, repaired.global_features)
    assert cached.image_ids == ("different-id",)


@pytest.mark.parametrize("factory", [StructuralEncoder, StructuralFeatureEncoder])
def test_general_solver_accepts_both_structural_encoder_contracts(factory):
    encoder = factory(side=64)
    solver = GeneralVisualSolver(structural_encoder=encoder)
    pictures, ids = [_solid(30), _solid(210)], ("second", "first")
    batch = solver.encode(pictures, ids)
    assert isinstance(batch, FeatureBatch)
    assert batch.image_ids == ids
    assert batch.global_features.shape == (2, 1088)
    assert "topology" in batch.views()
    assert "validity_indicators" not in batch.metadata["automatic_feature_views"]


def test_shared_structural_encoder_runs_support_fit_and_unseen_prediction():
    solver = GeneralVisualSolver(structural_encoder=StructuralFeatureEncoder(side=64))
    result = solver.solve([_solid(20), _solid(30), _solid(40)],
                          [_solid(210), _solid(220), _solid(230)], [_solid(35)])
    assert result.support_ids == ("A01", "A02", "A03", "B01", "B02", "B03")
    assert result.query_ids == ("Q01",) and len(result.predictions) == 1
    assert result.predictions[0].probability is None
    assert result.diagnostics["query_labels_used"] is False
    assert result.diagnostics["logical_explanation"] is None

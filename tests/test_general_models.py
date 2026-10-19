import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.reasoning.concepts import ConceptInducer, Concept, support_sheet
from src.reasoning.general import GeneralVisualSolver, validate_episode_images
from src.reasoning.features import FeatureBatch
from src.vision.pretrained import SiglipEncoder, LocalModelUnavailable, image_digest, model_directory


def image(value):
    return np.full((12, 12, 3), value, np.uint8)


class Backend:
    model_id = "fake-local"
    def __init__(self): self.calls = []
    def complete(self, prompt, images, *, max_new_tokens):
        self.calls.append((prompt, images))
        if "hypotheses" in prompt:
            return json.dumps({"hypotheses": [{"positive": "dark image", "negative": "light image"}]})
        dark = np.asarray(images[0]).mean() < 128
        return json.dumps({"label": "A" if dark else "B", "evidence": "pixel intensity"})


def test_concept_proposal_does_not_receive_queries_and_verification_hides_group():
    backend = Backend()
    positive, negative = [image(20), image(30)], [image(220), image(230)]
    query = image(40)
    result = ConceptInducer(backend).solve(positive, negative, [query])
    assert len(backend.calls) == 6
    assert len(backend.calls[0][1]) == 1  # support contact sheet only
    assert backend.calls[0][1][0].height > 12
    assert result.candidates[0].consistent
    assert result.predictions[0].label is True
    assert all(len(images) == 1 and "support" not in prompt.lower() for prompt, images in backend.calls[1:])
    assert result.diagnostics["query_images_used_in_proposal"] is False
    assert result.diagnostics["support_consistency_is_heldout_accuracy"] is False


def test_failed_support_concept_abstains_without_query_evaluation():
    backend = Backend()
    result = ConceptInducer(backend).solve([image(20), image(220)], [image(210), image(230)], [image(40)])
    assert not result.candidates[0].consistent
    assert result.predictions[0].label is None
    assert len(backend.calls) == 5


def test_model_invalid_output_is_unknown_and_never_executed():
    backend = Backend()
    backend.complete = lambda *args, **kwargs: '__import__("os").system("echo bad")'
    judgment = ConceptInducer(backend).judge(Concept("dark", "light"), image(30))
    assert judgment.label is None
    with pytest.raises(ValueError): ConceptInducer(backend).propose([image(30)], [image(200)])


def test_contradictory_and_duplicate_queries_are_not_silently_used_as_validation():
    with pytest.raises(ValueError, match="duplicates"):
        validate_episode_images([image(20), image(30)], [image(210), image(220)], [image(20)])
    with pytest.raises(ValueError, match="at least two"):
        validate_episode_images([image(20)], [image(210), image(220)])


def test_optional_encoder_missing_weights_never_downloads(tmp_path):
    assert model_directory("siglip", tmp_path) is None
    encoder = SiglipEncoder(cache_dir=tmp_path, feature_cache=None)
    with pytest.raises(LocalModelUnavailable, match="weights missing"):
        encoder.encode([image(40)])


def test_feature_cache_is_pixel_and_revision_based_not_label_based(tmp_path):
    encoder = SiglipEncoder(cache_dir=tmp_path, feature_cache=tmp_path/"features")
    picture = image(50)
    digest = image_digest(picture)
    path = encoder._cache_path(digest)
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{"global": np.ones(768), "patches": np.ones((196, 768)),
                                      "encoder_id": encoder.encoder_id})
    batch = encoder.encode([picture], ["anonymous"])
    assert batch.image_ids == ("anonymous",)
    assert batch.global_features.shape == (1, 768)
    assert encoder._model is None
    assert image_digest(Image.fromarray(picture)) == digest


def test_general_solver_returns_classifier_and_no_invented_rule():
    class Encoder:
        def encode(self, images, image_ids):
            values = np.array([[np.asarray(i).mean()/255, 1] for i in images])
            return FeatureBatch(tuple(image_ids), values, encoder_id="test")
    solver = GeneralVisualSolver(semantic_encoder=Encoder())
    result = solver.solve([image(20), image(25), image(30)], [image(210), image(220), image(230)], [image(40)])
    assert result.diagnostics["logical_explanation"] is None
    assert result.diagnostics["query_labels_used"] is False
    assert len(result.predictions) == 1
    assert result.predictions[0].probability is None
    assert result.to_dict()["encoder_id"] == "test"

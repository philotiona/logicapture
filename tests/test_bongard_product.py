from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.reasoning.bongard import infer_problem
from src.reasoning.episode import EpisodeClassifier

FIXTURES = Path(__file__).resolve().parents[1]/"data/fixtures/arrow_paths"


def groups(count=3):
    return [tuple(np.array(Image.open(FIXTURES/f"{letter}{i:02d}.png").convert("RGB"))
                  for i in range(1, count+1)) for letter in "AB"]


def test_product_rejects_small_groups_and_repeated_or_conflicting_drawings():
    positive, negative = groups()
    with pytest.raises(ValueError, match="at least three"):
        infer_problem(positive[:2], negative)
    with pytest.raises(ValueError, match="repeats"):
        infer_problem((positive[0],)*3, negative)
    with pytest.raises(ValueError, match="both groups"):
        infer_problem(positive, positive)
    with pytest.raises(ValueError, match="at most eight"):
        infer_problem(positive*3, negative)


def test_unreadable_drawing_domain_never_falls_back_to_feature_classification(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Classification must not substitute for a recovered Bongard rule")
    monkeypatch.setattr(EpisodeClassifier, "fit", fail)
    positive = [np.full((48, 48, 3), (150+index*10, 10, 20), np.uint8) for index in range(3)]
    negative = [np.full((48, 48, 3), (10, 20, 150+index*10), np.uint8) for index in range(3)]
    result = infer_problem(positive, negative)
    assert result.model is None and result.examples == ()
    assert result.to_dict()["rule_found"] is False
    assert result.diagnostics["logical_explanation"] is None


def test_program_and_generated_pixels_are_bound_to_all_given_examples():
    positive, negative = groups()
    result = infer_problem(positive, negative)
    assert result.model is not None
    assert result.model.support_ids == ("A01", "A02", "A03", "B01", "B02", "B03")
    assert len(result.diagnostics["structural_attempt"]["support_evidence"]) == 6
    assert result.diagnostics["all_given_examples_used"]
    assert result.diagnostics["query_labels_used"] is False
    assert [example.label for example in result.examples] == [True, False]
    for example in result.examples:
        assert example.verification["pixel_reread"]
        assert not any(np.array_equal(example.image, source) for source in positive+negative)
        assert all(vote is example.label for vote in example.verification["rule_votes"])
    predictions = result.model.predict([example.image for example in result.examples]).predictions
    assert [prediction.label for prediction in predictions] == [True, False]

import cv2
import numpy as np
import pytest

from src.reasoning.local_lab import NEAR_COPY, _silhouette, _similarity, lab_demo, learn_and_draw


def test_lab_draws_a_new_example_and_scores_independent_renderings():
    positive, negative = lab_demo()
    result = learn_and_draw(positive, negative)
    assert result.candidate is not None and result.decision.label is True
    assert result.donors[0] != result.donors[1]
    assert result.similarity < NEAR_COPY
    assert cv2.connectedComponents(np.uint8(result.candidate[:, :, 0] < 200))[0]-1 == 3

    for seed in (33, 77):
        fresh_a, fresh_b = lab_demo(seed=seed)
        predictions = [result.check(image).label for image in fresh_a + fresh_b]
        assert predictions == [True]*len(fresh_a) + [False]*len(fresh_b)
    with pytest.raises(ValueError, match="repeats a reference"):
        result.check(positive[0])


def test_lab_rejects_repeated_inputs_and_aligned_copies():
    positive, _ = lab_demo()
    with pytest.raises(ValueError, match="distinct drawings"):
        learn_and_draw(positive[:3], positive[:3])

    mark = np.full((128, 128, 3), 255, np.uint8)
    moved = mark.copy()
    mark[16:44, 20:52] = 0
    moved[51:79, 40:72] = 0
    assert _similarity(_silhouette(mark), _silhouette(moved)) >= NEAR_COPY

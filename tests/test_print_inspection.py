import cv2
import numpy as np
from PIL import ImageDraw
import pytest

from src.interface.print_inspector import demo_photo
from src.vision.print_inspection import demo_card, inspect_card, learn_reference, RetakePhoto


@pytest.mark.parametrize("seed", [7, 120, 250])
@pytest.mark.parametrize(("kind", "count", "issue"), [
    ("good", 5, None), ("missing", 4, "missing"), ("extra", 6, "extra"), ("misplaced", 5, "misplaced"),
])
def test_angled_and_rotated_card_photos(seed, kind, count, issue):
    reference = learn_reference(np.array(demo_card()))
    photo = np.rot90(demo_photo(kind, seed=seed), seed % 4).copy()
    result = inspect_card(photo, reference)
    assert (result.expected, result.observed) == (5, count)
    assert [defect.kind for defect in result.defects] == ([issue] if issue else [])
    assert result.passed is (issue is None)


def test_learns_an_accepted_photo_of_a_custom_layout():
    accepted = demo_card("blank")
    draw = ImageDraw.Draw(accepted)
    draw.ellipse((333, 226, 365, 258), fill="black")
    draw.rectangle((628, 378, 659, 409), fill="black")
    reference = learn_reference(np.array(accepted))
    assert inspect_card(np.rot90(np.array(accepted)).copy(), reference).passed
    draw.rectangle((480, 455, 500, 474), fill="black")
    result = inspect_card(np.array(accepted), reference)
    assert result.expected == 2 and result.observed == 3
    assert result.defects[0].kind == "extra"
    assert result.defects[0].observed.center == pytest.approx((490, 465), abs=2)


def test_retake_unregistered_or_too_small_photos():
    accepted = np.array(demo_card())
    reference = learn_reference(accepted)
    with pytest.raises(RetakePhoto, match="four corner markers"):
        inspect_card(accepted[:, 150:], reference)
    with pytest.raises(RetakePhoto):
        inspect_card(cv2.resize(accepted, (230, 154)), reference)
    with pytest.raises(ValueError, match="at least one"):
        learn_reference(np.array(demo_card("blank")))

import numpy as np
from PIL import Image, ImageDraw

from src.reasoning.grounding import MeasurementGrounding


def drawing(radius, offset=0):
    image = Image.new("RGB", (120, 120), "white")
    ImageDraw.Draw(image).ellipse((60-radius+offset, 60-radius, 60+radius+offset, 60+radius), outline="black", width=3)
    return np.array(image)


def test_grounding_selects_support_contrasts_without_using_query_images():
    a, b = [drawing(r) for r in (30, 34, 38)], [drawing(r) for r in (5, 7, 9)]
    context = MeasurementGrounding().fit(a, b)
    assert context.names
    assert "Measurements:" in context.support_text
    before = context.names, context.support_text
    description = context.describe(drawing(20, offset=10))
    assert "no group label" in description
    assert (context.names, context.support_text) == before
    inverted = MeasurementGrounding().fit(b, a)
    assert inverted.names == context.names


def test_photo_like_supports_disable_ink_measurement_grounding():
    rng = np.random.default_rng(17)
    images = [rng.integers(0, 256, (120, 120, 3), dtype=np.uint8) for _ in range(6)]
    context = MeasurementGrounding().fit(images[:3], images[3:])
    assert context.names == ()
    assert context.describe(images[0]) == ""

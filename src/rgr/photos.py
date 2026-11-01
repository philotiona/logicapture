"""Bounded photo decoding and explicit, reproducible manual crops."""
from io import BytesIO
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.vision.pretrained import rgb_image

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 16_000_000


def decode_photo(data):
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise ValueError("Use an image file no larger than 15 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.width*image.height > MAX_PIXELS or min(image.size) < 8:
                    raise ValueError("Use an image at least 8 × 8 pixels and at most 16 megapixels.")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Use a still image; animated images cannot be tested as one photograph.")
                return np.array(rgb_image(image))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Could not decode this image. Use a PNG, JPEG, WebP or BMP photograph.") from error


def crop_photo(image, horizontal=(0, 100), vertical=(0, 100)):
    """Crop original pixels using percentage bounds; no tone correction."""
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("A photo crop requires RGB uint8 pixels.")
    for limits in (horizontal, vertical):
        if (not isinstance(limits, (tuple, list, np.ndarray)) or len(limits) != 2
                or any(not isinstance(value, (int, float, np.number))
                                  or not np.isfinite(value) for value in limits)
                or not 0 <= limits[0] < limits[1] <= 100):
            raise ValueError("Crop limits must increase between 0 and 100 percent.")
    height, width = array.shape[:2]
    box = (round(width*horizontal[0]/100), round(height*vertical[0]/100),
           round(width*horizontal[1]/100), round(height*vertical[1]/100))
    left, top, right, bottom = box
    if right-left < 8 or bottom-top < 8:
        raise ValueError("Keep at least 8 pixels in each crop dimension.")
    return array[top:bottom, left:right].copy(), box

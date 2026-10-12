import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(source):
    with Image.open(source) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("Image exceeds 20 megapixels; resize before uploading")
        image = ImageOps.exif_transpose(image)
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image.convert("RGBA"))
        image = background.convert("RGB")
        if max(image.size) > 2400:
            image.thumbnail((2400, 2400))
        return np.array(image)


def _order(corners):
    points = np.array(corners, dtype=np.float32).reshape(4, 2)
    # Angular ordering remains well defined for diamond-shaped pages.
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    return points


def rectify_page(image, corners=None):
    """Rectify a visible page boundary; explicit corners override detection.

    Returns (RGB image, applied). Arbitrary page/marker detection is not promised.
    """
    if corners is None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = image.shape[0]*image.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            approx = cv2.approxPolyDP(contour, .025*cv2.arcLength(contour, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > .55*area:
                corners = approx.reshape(4, 2)
                break
    if corners is None: return image, False
    points = _order(corners)
    if not cv2.isContourConvex(points.reshape(4, 1, 2)) or cv2.contourArea(points) < 100:
        raise ValueError("Page corners must define a nondegenerate convex quadrilateral")
    h, w = image.shape[:2]
    if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
        raise ValueError("Page corners lie outside the image")
    tl, tr, br, bl = points
    width = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
    height = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
    target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    transform = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(image, transform, (width, height), borderValue=(255, 255, 255)), True

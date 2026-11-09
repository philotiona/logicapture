"""Register a printed card and compare its dark marks with a reference layout."""
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIZE = (1080, 720)
INSPECTION = (172, 166, 908, 554)
MARKER_SIDE = 96
MARKERS = {7: (48, 48), 18: (936, 48), 32: (936, 576), 45: (48, 576)}
MIN_MARK_AREA = 65
POSITION_TOLERANCE = 20


class RetakePhoto(ValueError):
    """The photo cannot support an inspection verdict."""


@dataclass(frozen=True)
class Mark:
    box: tuple[int, int, int, int]
    center: tuple[float, float]
    area: int


@dataclass(frozen=True)
class Defect:
    kind: str
    expected: Mark | None = None
    observed: Mark | None = None


@dataclass(frozen=True)
class Reference:
    marks: tuple[Mark, ...]


@dataclass(frozen=True)
class Inspection:
    image: np.ndarray
    expected: int
    observed: int
    defects: tuple[Defect, ...]

    @property
    def passed(self):
        return not self.defects


@lru_cache(maxsize=1)
def _dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def demo_card(kind="good"):
    if kind not in {"good", "missing", "extra", "misplaced", "blank"}:
        raise ValueError("Unknown print demo card.")
    image = Image.new("RGB", SIZE, "white")
    draw = ImageDraw.Draw(image)
    for marker, point in MARKERS.items():
        patch = cv2.aruco.generateImageMarker(_dictionary(), marker, MARKER_SIDE)
        image.paste(Image.fromarray(patch), point)
    draw.rectangle((153, 150, 927, 569), outline="#c9cdd3", width=2)
    if kind == "blank":
        return image
    for index, (x, y) in enumerate(((287, 252), (540, 252), (793, 252), (287, 466), (793, 466))):
        if kind == "missing" and index == 1:
            continue
        if kind == "misplaced" and index == 4:
            x -= 76
        if index in (0, 3):
            draw.ellipse((x-22, y-22, x+22, y+22), fill="black")
        elif index == 2:
            draw.polygon([(x, y-23), (x-24, y+21), (x+24, y+21)], fill="black")
        else:
            draw.rectangle((x-21, y-21, x+21, y+21), fill="black")
    if kind == "extra":
        draw.rectangle((534, 447, 546, 485), fill="black")
        draw.rectangle((521, 460, 559, 472), fill="black")
    return image


def register_card(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(_dictionary(), parameters).detectMarkers(gray)
    detections = [(int(marker), points.reshape(4, 2)) for marker, points in zip(
        ids.ravel() if ids is not None else (), corners) if int(marker) in MARKERS]
    if len(detections) != len(MARKERS) or {marker for marker, _ in detections} != set(MARKERS):
        raise RetakePhoto("Show all four corner markers on one flat card and retake the photo.")
    source, target = [], []
    for marker, points in detections:
        if min(np.linalg.norm(points-np.roll(points, 1, axis=0), axis=1)) < 24:
            raise RetakePhoto("Move closer so the card and all four markers are sharp.")
        x, y = MARKERS[marker]
        span = MARKER_SIDE-1
        source.extend(points)
        target.extend(((x, y), (x+span, y), (x+span, y+span), (x, y+span)))
    source, target = np.float32(source), np.float32(target)
    transform, _ = cv2.findHomography(source, target)
    if transform is None or not np.isfinite(transform).all():
        raise RetakePhoto("The corner markers could not be aligned.")
    projected = cv2.perspectiveTransform(source[None], transform)[0]
    if np.max(np.linalg.norm(projected-target, axis=1)) > 4.5:
        raise RetakePhoto("The card looks bent or out of focus. Flatten it and retake the photo.")
    result = cv2.warpPerspective(image, transform, SIZE, borderValue=(255, 255, 255))
    for marker, (x, y) in MARKERS.items():
        patch = cv2.cvtColor(result[y:y+MARKER_SIDE, x:x+MARKER_SIDE], cv2.COLOR_RGB2GRAY)
        pattern = cv2.aruco.generateImageMarker(_dictionary(), marker, MARKER_SIDE)
        interior = cv2.erode(pattern, np.ones((5, 5), np.uint8)) > 0
        dark = cv2.erode(255-pattern, np.ones((5, 5), np.uint8)) > 0
        if np.median(patch[interior]) - np.median(patch[dark]) < 65:
            raise RetakePhoto("The markers have low contrast. Avoid glare and use brighter, even light.")
    return result


def _marks(rectified):
    left, top, right, bottom = INSPECTION
    gray = cv2.cvtColor(rectified[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
    # Estimate paper brightness near each mark, so a gradual shadow does not
    # become a defect. The demonstration uses small isolated dark stamps.
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (63, 63)))
    ink = (cv2.subtract(paper, gray) >= 29) & (gray < .80*paper)
    if np.mean(ink) > .32:
        raise RetakePhoto("The inspection area is mostly dark. Check the lighting and card layout.")
    mask = cv2.morphologyEx(np.uint8(ink), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    marks = []
    for (x, y, w, h, area), (cx, cy) in zip(stats[1:], centers[1:]):
        if area < MIN_MARK_AREA:
            continue
        marks.append(Mark((int(x)+left, int(y)+top, int(x+w)+left, int(y+h)+top),
                          (float(cx)+left, float(cy)+top), int(area)))
    if len(marks) > 30:
        raise RetakePhoto("Too many marks or shadows were detected. Use a clean card with up to 30 separate marks.")
    return tuple(marks)


def learn_reference(image):
    card = register_card(image)
    marks = _marks(card)
    if not marks:
        raise ValueError("The reference needs at least one clear dark mark between the corner markers.")
    return Reference(marks)


def _distance(left, right):
    return float(np.hypot(left.center[0]-right.center[0], left.center[1]-right.center[1]))


def _pair_marks(expected, observed, tolerance, compatible):
    pairs = sorted((distance, i, j) for i, left in expected.items() for j, right in observed.items()
                   if (distance := _distance(left, right)) <= tolerance and compatible(left, right))
    for _, i, j in pairs:
        if i in expected and j in observed:
            yield expected.pop(i), observed.pop(j)


def inspect_card(image, reference):
    card = register_card(image)
    marks = _marks(card)
    expected, observed = dict(enumerate(reference.marks)), dict(enumerate(marks))
    defects = []
    for left, right in _pair_marks(expected, observed, POSITION_TOLERANCE, lambda *_: True):
        if not .58 <= right.area/left.area <= 1.52:
            defects.append(Defect("changed", left, right))
    for left, right in _pair_marks(expected, observed, 135, lambda a, b: .75 <= b.area/a.area <= 1.33):
        defects.append(Defect("misplaced", left, right))
    defects.extend(Defect("missing", expected=mark) for mark in expected.values())
    defects.extend(Defect("extra", observed=mark) for mark in observed.values())
    return Inspection(card, len(reference.marks), len(marks), tuple(defects))


def annotate(result):
    image = Image.fromarray(result.image)
    draw = ImageDraw.Draw(image)
    color = "#17834b" if result.passed else "#c23431"
    draw.rectangle(INSPECTION, outline=color, width=2)
    font = ImageFont.load_default(size=21)
    for defect in result.defects:
        mark = defect.observed or defect.expected
        x, y, right, bottom = mark.box
        draw.rectangle((x-5, y-5, right+5, bottom+5), outline=color, width=4)
        if defect.kind == "misplaced":
            draw.line((defect.expected.center, defect.observed.center), fill=color, width=4)
            draw.rectangle(defect.expected.box, outline=color, width=2)
        label = defect.kind.capitalize()
        width = int(draw.textlength(label, font=font))
        draw.rectangle((x-4, y-32, x+width+8, y-7), fill=color)
        draw.text((x+1, y-34), label, fill="white", font=font)
    return image

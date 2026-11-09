"""Split a framed example sheet without using its labels, colors, or caption."""
import cv2
import numpy as np


def split_example_sheet(image):
    """Return the left references, right references, and unlabeled query crops."""
    error = ("Could not find a complete example sheet. Use an upright page with six framed "
             "references on each side in three rows and two framed test images below.")
    height, width = image.shape[:2]
    scale = min(1.0, 1600 / max(height, width))
    if min(height, width) * scale < 120:
        raise ValueError(error)
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    h, w = preview.shape[:2]
    gray = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 35, 100)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if (max(28, .10*w) < cw < .34*w and max(28, .06*h) < ch < .31*h
                and .67 < cw/ch < 1.5 and cv2.contourArea(contour) > .86*cw*ch):
            candidates.append((x, y, cw, ch))
    boxes = []
    for x, y, cw, ch in sorted(candidates, key=lambda box: -box[2]*box[3]):
        cx, cy = x+cw/2, y+ch/2
        # Keep the outer edge when both sides of a frame were detected.
        if not any(bx < cx < bx+bw and by < cy < by+bh for bx, by, bw, bh in boxes):
            boxes.append((x, y, cw, ch))

    if len(boxes) != 14:
        raise ValueError(error)
    cell_width, cell_height = np.median(np.array(boxes)[:, 2:], axis=0)
    if any(abs(cw-cell_width) > .18*cell_width or abs(ch-cell_height) > .18*cell_height
           for _, _, cw, ch in boxes):
        raise ValueError(error)

    rows = []
    for box in sorted(boxes, key=lambda box: box[1]+box[3]/2):
        center_y = box[1]+box[3]/2
        if not rows or center_y - (rows[-1][0][1]+rows[-1][0][3]/2) > .20*cell_height:
            rows.append([])
        rows[-1].append(box)
    if [len(row) for row in rows] != [4, 4, 4, 2]:
        raise ValueError(error)
    for row in rows:
        row.sort(key=lambda box: box[0])
        if any(left[0]+left[2] > right[0] for left, right in zip(row, row[1:])):
            raise ValueError(error)
    for above, below in zip(rows, rows[1:]):
        if max(y+ch for _, y, _, ch in above) > min(y for _, y, _, _ in below):
            raise ValueError(error)
    if (any(row[1][0]+row[1][2] > w/2 or row[2][0] < w/2 for row in rows[:3])
            or any(abs(box[0]-first[0]) > .16*cell_width
                   for row in rows[1:3] for box, first in zip(row, rows[0]))):
        raise ValueError(error)

    def crop(box):
        x, y, cw, ch = box
        # Remove the printed frame and JPEG bleed on every side. Extract from
        # the source image so the downloaded test tiles retain its detail.
        inset = max(3, round(.035 * min(cw, ch)))
        left, right = (round(px * width/w) for px in (x+inset, x+cw-inset))
        top, bottom = (round(py * height/h) for py in (y+inset, y+ch-inset))
        return image[top:bottom, left:right]

    return (tuple(crop(box) for row in rows[:3] for box in row[:2]),
            tuple(crop(box) for row in rows[:3] for box in row[2:]),
            tuple(crop(box) for box in rows[3]))

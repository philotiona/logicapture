from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Panel:
    image: np.ndarray
    label: bool
    box: tuple[int, int, int, int]


def detect_panels(image, divider=None, expected_per_group=6):
    h, w = image.shape[:2]
    divider = w/2 if divider is None else divider
    if not 0 < divider < w: raise ValueError("Divider must lie inside the page")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not .025*w*h < area < .18*w*h: continue
        approx = cv2.approxPolyDP(contour, .018*cv2.arcLength(contour, True), True)
        x, y, cw, ch = cv2.boundingRect(contour)
        if len(approx) == 4 and cv2.isContourConvex(approx) and .65 < cw/ch < 1.5 and area/(cw*ch) > .87:
            boxes.append((x, y, cw, ch))
    # Outer/inner borders describe one cell; keep the larger one.
    kept = []
    for box in sorted(boxes, key=lambda b: -b[2]*b[3]):
        x, y, cw, ch = box
        if any(abs(x+cw/2-a-c/2) < .12*c and abs(y+ch/2-b-d/2) < .12*d for a, b, c, d in kept): continue
        kept.append(box)
    panels = []
    for x, y, cw, ch in sorted(kept, key=lambda b: (b[0]+b[2]/2 >= divider, b[1]//max(1, h//20), b[0])):
        inset = max(5, round(.027*min(cw, ch)))
        panels.append(Panel(image[y+inset:y+ch-inset, x+inset:x+cw-inset].copy(), x+cw/2 < divider, (x, y, cw, ch)))
    counts = [sum(p.label == label for p in panels) for label in (True, False)]
    if counts != [expected_per_group, expected_per_group]:
        raise ValueError(f"Detected {counts[0]} Group A and {counts[1]} Group B cells; expected {expected_per_group} each. Use the supplied template, straighten the photograph, adjust the divider, or upload each cell separately.")
    return panels


def panel_overlay(image, panels):
    out = image.copy()
    for i, p in enumerate(panels):
        x, y, w, h = p.box
        color = (20, 150, 110) if p.label else (219, 112, 53)
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 3)
        cv2.putText(out, f"{'A' if p.label else 'B'} {i%6+1}", (x+8, y+22), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    return out

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Panel:
    image: np.ndarray
    label: bool
    box: tuple[int, int, int, int]
    corners: tuple = ()
    inferred: bool = False


def detect_panels(image, divider=None, expected_per_group=6):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 24:
        raise ValueError("Use an RGB page at least 24 pixels on each side")
    h, w = image.shape[:2]
    divider = w/2 if divider is None else divider
    if not 0 < divider < w: raise ValueError("Divider must lie inside the page")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not .005*w*h < area < .24*w*h: continue
        approx = cv2.approxPolyDP(contour, .018*cv2.arcLength(contour, True), True)
        x, y, cw, ch = cv2.boundingRect(contour)
        if len(approx) == 4 and cv2.isContourConvex(approx) and .55 < cw/ch < 1.8 and area/(cw*ch) > .70 and min(cw, ch) >= 28:
            boxes.append(((x, y, cw, ch), approx.reshape(4, 2)))
    # Outer/inner borders describe one cell; keep the larger one.
    kept = []
    for box, corners in sorted(boxes, key=lambda b: -b[0][2]*b[0][3]):
        x, y, cw, ch = box
        if any(cv2.pointPolygonTest(q.astype(np.float32), (x+cw/2, y+ch/2), False) >= 0
               for _, q in kept): continue
        kept.append((box, corners))
    panels = []
    # Cluster row centers by cell scale rather than arbitrary page-height bins.
    for label in (True, False):
        group = [(box, q) for box, q in kept if (box[0]+box[2]/2 < divider) == label]
        rows = []
        for item in sorted(group, key=lambda item: item[0][1]+item[0][3]/2):
            box, q = item
            cy = box[1]+box[3]/2
            if not rows or cy-np.mean([b[1]+b[3]/2 for b, _ in rows[-1]]) > .45*box[3]:
                rows.append([])
            rows[-1].append(item)
        for row in rows:
            for (x, y, cw, ch), corners in sorted(row, key=lambda item: item[0][0]):
                from .page_rectification import rectify_page
                edges = np.roll(corners,-1,axis=0)-corners
                axial = np.all(np.min(np.abs(edges),axis=1)/np.maximum(1,np.linalg.norm(edges,axis=1)) < .09)
                crop = image[y:y+ch,x:x+cw] if axial else rectify_page(image,corners)[0]
                inset = max(5, round(.035*min(crop.shape[:2])))
                crop = crop[inset:-inset, inset:-inset].copy()
                panels.append(Panel(crop, label, (x, y, cw, ch), tuple(map(tuple, corners.tolist()))))
    counts = [sum(p.label == label for p in panels) for label in (True, False)]
    # A single broken/missing border can be recovered from eleven observed
    # cells in a consistent 2×3 layout. Never invent an entire unobserved grid.
    if sorted(counts) == [5,6] and expected_per_group in (None,6):
        complete_label = counts[0] == 6
        complete = [p for p in panels if p.label == complete_label]
        partial = [p for p in panels if p.label != complete_label]
        widths = np.array([p.box[2] for p in panels])
        heights = np.array([p.box[3] for p in panels])
        mw,mh = float(np.median(widths)),float(np.median(heights))
        if widths.std()/mw < .08 and heights.std()/mh < .08:
            ys = sorted([p.box[1] for p in complete])
            rows = [float(np.mean(ys[i:i+2])) for i in (0,2,4)]
            xs = sorted([p.box[0] for p in partial])
            gaps = np.diff(xs)
            split = int(np.argmax(gaps))+1
            cols = [float(np.median(xs[:split])),float(np.median(xs[split:]))]
            regular = (min(np.diff(rows)) > .8*mh and cols[1]-cols[0] > .8*mw
                       and all(min(abs(p.box[1]-y) for y in rows) < .08*mh for p in partial)
                       and all(min(abs(p.box[0]-x) for x in cols) < .08*mw for p in partial))
            missing = [(round(x),round(y)) for y in rows for x in cols
                       if not any(abs(p.box[0]-x)<.1*mw and abs(p.box[1]-y)<.1*mh for p in partial)]
            if regular and len(missing)==1:
                x,y = missing[0]
                cw,ch = round(mw),round(mh)
                inset = max(5,round(.035*min(cw,ch)))
                if x>=0 and y>=0 and x+cw<=w and y+ch<=h:
                    panels.append(Panel(image[y+inset:y+ch-inset,x+inset:x+cw-inset].copy(),not complete_label,
                                        (x,y,cw,ch),(),True))
                    panels.sort(key=lambda p:(not p.label,round(p.box[1]/mh),p.box[0]))
                    counts = [6,6]
    expected = expected_per_group
    valid = counts == [expected, expected] if expected is not None else counts[0] == counts[1] and 2 <= counts[0] <= 20
    if not valid:
        wanted = f"{expected} each" if expected is not None else "equal groups of 2–20 framed panels"
        raise ValueError(f"Detected {counts[0]} positive and {counts[1]} negative panels; need {wanted}. Keep all panel borders visible and the positive group on the left. Adjust the page divider if needed.")
    return panels


def panel_overlay(image, panels):
    out = image.copy()
    counts = {True: 0, False: 0}
    for p in panels:
        x, y, w, h = p.box
        color = (0, 47, 167)
        counts[p.label] += 1
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 3)
        cv2.putText(out, f"{'A' if p.label else 'B'}{counts[p.label]:02d}", (x+8, y+22), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    return out


def split_problem_page(image, *, divider=.5, expected_per_group=None, rectify=True):
    """Shared upload/camera path. Return the corrected page and checked crops."""
    from .page_rectification import rectify_page
    if not 0 < divider < 1:
        raise ValueError("Page divider must be between zero and one")
    page, applied = rectify_page(image) if rectify else (image, False)
    panels = detect_panels(page, divider=page.shape[1]*divider, expected_per_group=expected_per_group)
    return page, panels, {"rectified": applied, "positive": sum(p.label for p in panels),
                          "negative": sum(not p.label for p in panels), "inferred_panels":sum(p.inferred for p in panels),
                          "method": "observed_panel_frames_with_one_grid_repair" if any(p.inferred for p in panels) else "observed_panel_frames"}

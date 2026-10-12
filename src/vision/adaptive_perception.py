"""Evidence-first interpretation of ink as regions and ordered stroke paths.

Segmentation is independent of the group labels. Region recognition remains an
optional view of a closed boundary; an open stroke is never a polygon merely
because its ink envelope happens to have three corners.
"""
from dataclasses import replace
import cv2
import numpy as np

from src.scene import Scene, VisualObject, GeometryTolerance, derive_relations


def ink_mask(image, adaptive=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if adaptive:
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 10)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def remove_ruling(mask):
    """Remove isolated thin panel frames, including a bottom/right L ruling.

Only disconnected components made almost entirely of long straight edges in
the outer fifth qualify. A stroke touching a frame stays unresolved instead
of being silently erased. No panel index or group label is used.
"""
    h, w = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean, removed = mask.copy(), 0
    for index in range(1, count):
        x, y, cw, ch, area = stats[index]
        if cw < .6*w or ch < .6*h or area > .10*w*h:
            continue
        component = np.uint8(labels == index)*255
        horizontal = cv2.morphologyEx(component, cv2.MORPH_OPEN,
                                      np.ones((1, max(12, round(.55*w))), np.uint8))
        vertical = cv2.morphologyEx(component, cv2.MORPH_OPEN,
                                    np.ones((max(12, round(.55*h)), 1), np.uint8))
        yy, xx = np.where(horizontal)
        vy, vx = np.where(vertical)
        if not len(xx) or not len(vx):
            continue
        outer_h = (yy < .20*h) | (yy > .80*h)
        outer_v = (vx < .20*w) | (vx > .80*w)
        if np.mean(outer_h) < .95 or np.mean(outer_v) < .95:
            continue
        explained = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8))
        fraction = np.count_nonzero((explained > 0) & (component > 0))/area
        if fraction > .92:
            clean[labels == index] = 0
            removed += 1
    return clean, removed


def _component_for_object(mask, obj):
    h, w = mask.shape
    points = np.array([(round(x*w), round(y*h)) for x, y in obj.contour], np.int32)
    envelope = np.zeros_like(mask)
    cv2.fillPoly(envelope, [points], 255)
    # Include only the boundary's connected ink; exclude any nested region.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best, overlap = None, 0
    boundary = np.zeros_like(mask)
    cv2.polylines(boundary, [points], True, 255, 3)
    for i in range(1, count):
        score = np.count_nonzero((labels == i) & (boundary > 0))
        if score > overlap:
            best, overlap = i, score
    return np.uint8(labels == best)*255 if best is not None else mask & envelope


def extract_adaptive_scene(image, tolerance=GeometryTolerance(), adaptive=False,
                           min_area=.002, attribute_model=None, uncertain_only=True,
                           uncertainty_threshold=.96, remove_panel_ruling=True):
    from .object_detection import extract_geometric_scene
    from .stroke_graph import analyze_stroke
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 24:
        raise ValueError("Expected an RGB panel at least 24×24 pixels")
    h, w = image.shape[:2]
    raw = ink_mask(image, adaptive)
    mask, removed = remove_ruling(raw) if remove_panel_ruling else (raw, 0)
    clean_image = image.copy()
    clean_image[(raw > 0) & (mask == 0)] = 255
    scene = extract_geometric_scene(clean_image, tolerance, adaptive, min_area,
                                     attribute_model=None)
    # Preserve segmentation failures, but remove the legacy assumption that
    # every free contour represents a polygonal region.
    scene.warnings = [s for s in scene.warnings if not s.startswith("An irregular contour")]
    regions, neural_ids = [], []
    processed = np.zeros_like(mask)
    for index, obj in enumerate(scene.objects):
        component = _component_for_object(mask, obj)
        processed |= component
        ys, xs = np.where(component)
        if not len(xs):
            regions.append(np.full((48, 48), 255, np.uint8))
            continue
        cw, ch = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer = max(contours, key=cv2.contourArea)
        hull_area = max(1., cv2.contourArea(cv2.convexHull(outer)))
        all_contours, hierarchy = cv2.findContours(component, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        enclosed = any(hierarchy[0][i][3] >= 0 and cv2.contourArea(contour) > .2*hull_area
                       for i, contour in enumerate(all_contours)) if hierarchy is not None else False
        density = len(xs)/hull_area
        _, oriented_size, _ = cv2.minAreaRect(outer)
        slenderness = max(oriented_size)/max(1., min(oriented_size))
        # Thin boundaries carry a path view as well as optional region evidence.
        thin = not obj.filled or density < .52 or slenderness > 4
        path = analyze_stroke(component) if thin else None
        directed_marks = sum(arrow.reliable for arrow in path.arrows) if path is not None else 0
        open_stroke = path is not None and path.topology in ("open", "branched")
        known_region = obj.shape in ("circle", "triangle", "quadrilateral")
        # A closed free-form boundary keeps its region view (and its path view)
        # even when no primitive shape name applies. Containment must depend on
        # the observed boundary, not membership in the CNN's shape vocabulary.
        if thin and ((open_stroke and not enclosed) or directed_marks > 0 or (not known_region and not enclosed)):
            scene.objects[index] = replace(obj, shape="unknown", filled=False,
                kind="path", path=path, attribute_source="stroke_graph",
                confidence=path.confidence if path is not None else .0)
            if path is not None and path.topology == "unknown":
                scene.warnings.append(f"Path {obj.id}: traversal could not be resolved.")
        else:
            # Arbitrary closed regions are retained as unknown; CNN certainty
            # within its four training labels is not out-of-distribution evidence.
            shape = obj.shape if known_region else "unknown"
            scene.objects[index] = replace(obj, shape=shape, kind="region", path=path)
            if shape == "unknown":
                scene.warnings.append(f"Region {obj.id}: free-form boundary; shape class is unresolved.")
            elif attribute_model is not None and (not uncertain_only or obj.confidence < uncertainty_threshold):
                neural_ids.append(index)
        pad = max(3, round(.08*max(cw, ch)))
        isolated = np.where(component, 0, 255).astype(np.uint8)
        regions.append(isolated[max(0, ys.min()-pad):min(h, ys.max()+pad+1),
                                max(0, xs.min()-pad):min(w, xs.max()+pad+1)])
    if attribute_model is not None and neural_ids:
        from .attribute_cnn import apply_attribute_predictions
        apply_attribute_predictions(scene, regions, attribute_model, neural_ids, geometric_guard=True)
    # findContours measures enclosed area and can drop a perfectly valid thin
    # straight stroke whose envelope has zero area. Recover remaining ink by
    # connected-component ink length rather than manufacturing a polygon.
    remainder = mask & ~processed
    count, labels, stats, centers = cv2.connectedComponentsWithStats(remainder, 8)
    for i in range(1, count):
        x, y, cw, ch, ink_area = stats[i]
        if ink_area < max(12, min_area*w*h*.45) or max(cw, ch) < .08*min(w, h):
            continue
        component = np.uint8(labels == i)*255
        path = analyze_stroke(component)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = tuple((float(px)/w, float(py)/h) for px, py in max(contours, key=cv2.contourArea).reshape(-1, 2))
        scene.objects.append(VisualObject(len(scene.objects), "unknown", False,
            center_x=float(centers[i][0]/w), center_y=float(centers[i][1]/h),
            area=float(ink_area/(w*h)), contour=points, kind="path", path=path,
            attribute_source="stroke_graph", confidence=path.confidence if path is not None else 0.))
    if removed:
        scene.warnings.append(f"Excluded {removed} isolated panel ruling component(s).")
    # Geometric relations must be recomputed after classification. Paths keep
    # topology/direction on their evidence object instead of polygon containment.
    scene.relations.clear()
    return derive_relations(scene, tolerance)

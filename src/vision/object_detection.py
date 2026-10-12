import cv2
import numpy as np
from src.scene import Scene, VisualObject, GeometryTolerance, derive_relations


def extract_geometric_scene(image, tolerance=GeometryTolerance(), adaptive=False, min_area=.002,
                  attribute_model=None, uncertain_only=True, uncertainty_threshold=.96):
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 24:
        raise ValueError("Expected an RGB panel at least 24×24 pixels")
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if adaptive:
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    objects, warnings, regions, uncertain = [], [], [], []
    if hierarchy is None: return Scene()
    hierarchy = hierarchy[0]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for i, contour in enumerate(contours):
        depth, parent = 0, hierarchy[i][3]
        while parent != -1:
            depth += 1
            parent = hierarchy[parent][3]
        # Odd levels are the white interior of an outline, never a second object.
        if depth % 2: continue
        area = cv2.contourArea(contour)
        if area < min_area*w*h: continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x+cw >= w-1 or y+ch >= h-1:
            warnings.append("A contour touches the panel boundary and was excluded; check the crop.")
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, .035*perimeter, True)
        circularity = 4*np.pi*area/max(perimeter**2, 1)
        convex = cv2.isContourConvex(approx)
        if len(approx) == 3 and convex: shape, confidence = "triangle", .95
        elif len(approx) == 4 and convex: shape, confidence = "quadrilateral", .90
        elif circularity > .75 and len(approx) >= 5: shape, confidence = "circle", .90
        else:
            shape, confidence = "polygon", .45
            warnings.append("An irregular contour may be a free-form polygon or merged touching objects.")
        child, hole_area = hierarchy[i][2], 0
        while child != -1:
            hole_area += cv2.contourArea(contours[child])
            child = hierarchy[child][0]
        filled = hole_area / area < .35
        moments = cv2.moments(contour)
        if not moments["m00"]: continue
        cx, cy = moments["m10"]/moments["m00"]/w, moments["m01"]/moments["m00"]/h
        # Use ink pixels to classify color, avoiding white interiors and children.
        region = np.zeros_like(mask)
        cv2.drawContours(region, [contour], -1, 255, 2)
        ink = (region > 0) & (mask > 0)
        pixels = hsv[ink]
        color = "black"
        colored = pixels[pixels[:, 1] > 70]
        if len(colored) > .35*max(1, len(pixels)):
            hue = float(np.median(colored[:, 0]))
            color = "red" if hue < 15 or hue > 165 else "green" if hue < 85 else "blue"
        # Preserve actual outline for containment and evidence overlays.
        detailed = cv2.approxPolyDP(contour, .007*perimeter, True).reshape(-1, 2)
        points = tuple((float(px)/w, float(py)/h) for px, py in detailed)
        objects.append(VisualObject(len(objects), shape, bool(filled), color, cx, cy,
                                    min(area/(w*h), 1), float(cv2.minAreaRect(contour)[2]), points, confidence,
                                    attribute_source="opencv"))
        # Isolate this ink component, excluding hole interiors (including any
        # distinct nested objects). OpenCV owns segmentation; the CNN sees only
        # a normalized object region, never the problem labels or whole page.
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, -1)
        child = hierarchy[i][2]
        while child != -1:
            cv2.drawContours(component, [contours[child]], -1, 0, -1)
            child = hierarchy[child][0]
        isolated = np.full_like(gray, 255)
        isolated[(component > 0) & (mask > 0)] = 0
        pad = max(3, round(.08*max(cw, ch)))
        regions.append(isolated[max(0, y-pad):min(h, y+ch+pad), max(0, x-pad):min(w, x+cw+pad)])
        uncertain.append(confidence < uncertainty_threshold or abs(hole_area/area-.35) < .15)
    scene = Scene(objects, warnings=warnings)
    if attribute_model is not None and objects:
        from .attribute_cnn import apply_attribute_predictions
        ids = [i for i in range(len(objects)) if not uncertain_only or uncertain[i]]
        apply_attribute_predictions(scene, regions, attribute_model, ids)
    return derive_relations(scene, tolerance)


def extract_scene(image, tolerance=GeometryTolerance(), adaptive=False, min_area=.002,
                  attribute_model=None, uncertain_only=True, uncertainty_threshold=.96,
                  representation="adaptive", remove_panel_ruling=True):
    """Default evidence-first scene; legacy contour classifier is an ablation."""
    if representation == "geometric":
        return extract_geometric_scene(image, tolerance, adaptive, min_area, attribute_model,
                                        uncertain_only, uncertainty_threshold)
    if representation != "adaptive":
        raise ValueError("Representation must be adaptive or geometric")
    from .adaptive_perception import extract_adaptive_scene
    return extract_adaptive_scene(image, tolerance, adaptive, min_area, attribute_model,
                                  uncertain_only, uncertainty_threshold, remove_panel_ruling)


def detection_overlay(image, scene, evidence=None):
    out = image.copy()
    h, w = out.shape[:2]
    selected = set((evidence or {}).get("selected_ids", []))
    violating = set((evidence or {}).get("violating_ids", []))
    for obj in scene.objects:
        color = (0, 0, 0) if obj.id in violating else (0, 47, 167)
        points = np.array([(int(x*w), int(y*h)) for x, y in obj.vertices()], np.int32)
        if len(points) < 2:
            continue
        closed = obj.kind != "path" or (obj.path is not None and obj.path.topology == "closed")
        cv2.polylines(out, [points], closed, color, 2 if obj.id in selected or obj.id in violating else 1)
        if obj.kind == "path" and obj.path is not None:
            for index, arrow in enumerate(obj.path.arrows, start=1):
                if arrow.tip is None or arrow.tail is None: continue
                tip, tail = tuple(round(x*s) for x, s in zip(arrow.tip, (w, h))), tuple(round(x*s) for x, s in zip(arrow.tail, (w, h)))
                if arrow.reliable:
                    cv2.arrowedLine(out, tail, tip, color, 1, tipLength=.30)
                else:
                    cv2.circle(out, tip, 4, color, 1)
                cv2.putText(out, str(index), (max(2, tip[0]+4), max(12, tip[1]-4)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
        # IDs link to the inspector table. Long attribute labels obscured the
        # drawings, especially in the compact group overview.
        label_x, label_y = max(3, int(points[:, 0].min())), max(13, int(points[:, 1].min())-5)
        cv2.putText(out, str(obj.id), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    by_id = {o.id: o for o in scene.objects}
    for a, b in (evidence or {}).get("witnesses", []):
        oa, ob = by_id[a], by_id[b]
        cv2.arrowedLine(out, (int(oa.center_x*w), int(oa.center_y*h)), (int(ob.center_x*w), int(ob.center_y*h)), (0, 47, 167), 2)
    return out

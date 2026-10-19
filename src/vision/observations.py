"""Fast, named geometric observations, including relative dot proximity.

Measurements do not receive groups, filenames, or problem identities. Missing
evidence is None. Relative lengths use the observed marks, not the panel size.
"""
import cv2
import numpy as np

DESCRIPTIONS = {
    "component_count": "number of disconnected ink components",
    "hole_count": "number of enclosed white holes",
    "ink_fraction": "fraction of the panel covered by ink",
    "bbox_occupancy": "ink density inside the drawing's bounding box",
    "extent_x": "horizontal extent relative to the panel",
    "extent_y": "vertical extent relative to the panel",
    "mean_component_area": "mean component area relative to the panel",
    "median_solidity": "median component convexity (area / convex-hull area)",
    "minimum_solidity": "minimum component convexity (area / convex-hull area)",
    "median_circularity": "median component circularity",
    "area_variation": "variation in component sizes",
    "mirror_x": "agreement with a left-right reflection",
    "mirror_y": "agreement with a top-bottom reflection",
    "rotation_180": "agreement with a half-turn",
    "dot_spacing": "median nearest-dot spacing in dot-width units",
    "dot_spacing_p10": "10th-percentile nearest-dot spacing in dot-width units",
    "dot_spacing_p90": "90th-percentile nearest-dot spacing in dot-width units",
    "dot_spacing_variation": "variation in nearest-dot spacing",
}


def observations(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if max(gray.shape) > 768:
        h, w = gray.shape
        gray = cv2.resize(gray, (round(w*768/max(h,w)), round(h*768/max(h,w))))
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if gray.std() < .5:
        mask = np.uint8(gray < 128)*255
    # These are already cropped panels. Do not erase large rectangles as
    # presumed page ruling: their boundaries may be the concept being tested.
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    h, w = gray.shape
    keep = [i for i in range(1,count) if stats[i,4] >= max(3, .00004*w*h)]
    # A crop touching ink cannot establish its size, topology or spacing.
    clipped = [i for i in keep if stats[i,0] <= 0 or stats[i,1] <= 0 or stats[i,0]+stats[i,2] >= w
               or stats[i,1]+stats[i,3] >= h]
    # Partial marks cannot establish a total count, but complete interior dots
    # can still establish their local spacing. Never invent the missing count.
    keep = [i for i in keep if i not in clipped]
    clean = np.uint8(np.isin(labels, keep))*255
    values = {name: None for name in DESCRIPTIONS}
    values.update(component_count=None if clipped else float(len(keep)), ink_fraction=None if clipped else float(np.mean(clean > 0)), hole_count=None if clipped else 0.)
    if not keep:
        return values
    contours, hierarchy = cv2.findContours(clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not clipped:
        holes = 0
        for i,contour in enumerate(contours):
            parent,depth = hierarchy[0][i][3],0
            while parent != -1:
                depth += 1
                parent = hierarchy[0][parent][3]
            holes += depth%2 == 1 and cv2.contourArea(contour) >= 4
        values["hole_count"] = float(holes)
    yy, xx = np.where(clean)
    crop = clean[yy.min():yy.max()+1, xx.min():xx.max()+1] > 0
    values.update(bbox_occupancy=float(crop.mean()), extent_x=crop.shape[1]/w, extent_y=crop.shape[0]/h)
    areas = stats[keep,4].astype(float)
    values.update(mean_component_area=float(areas.mean()/(w*h)), area_variation=float(areas.std()/areas.mean()))
    for name, reflected in (("mirror_x", crop[:,::-1]), ("mirror_y", crop[::-1]), ("rotation_180", crop[::-1,::-1])):
        values[name] = float(np.count_nonzero(crop & reflected)/np.count_nonzero(crop | reflected))
    solidity, circularity = [], []
    for index in keep:
        contour = max(cv2.findContours(np.uint8(labels == index)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        solidity.append(area/max(1.,cv2.contourArea(cv2.convexHull(contour))))
        circularity.append(4*np.pi*area/max(1.,perimeter**2))
    values.update(median_solidity=float(np.median(solidity)), minimum_solidity=float(min(solidity)),
                  median_circularity=float(np.median(circularity)))
    dot_ids = [j for j in range(len(keep)) if .5*np.median(areas) <= areas[j] <= 2*np.median(areas)
               and circularity[j] > .4]
    if (3 <= len(dot_ids) <= 512 and len(dot_ids) >= .6*len(keep)
            and np.median(areas)/(w*h) < .04):
        points = centers[[keep[j] for j in dot_ids]]
        distances = np.linalg.norm(points[:,None]-points[None,:], axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)/np.sqrt(areas[dot_ids])
        values["dot_spacing"] = float(np.median(nearest))
        values["dot_spacing_p10"] = float(np.quantile(nearest,.1))
        values["dot_spacing_p90"] = float(np.quantile(nearest,.9))
        values["dot_spacing_variation"] = float(nearest.std()/nearest.mean())
    if clipped:
        for name in values:
            if not name.startswith("dot_spacing"):
                values[name] = None
    return values

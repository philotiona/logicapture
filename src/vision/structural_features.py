"""Label-free, numeric measurements of a drawing or image.

These measurements complement semantic image embeddings.  They do not name
objects, enumerate Bongard rules, fit an episode, or promise to understand an
arbitrary concept.  A learner can select useful dimensions from its support
examples.  Undefined measurements have a separate validity mask; a blank panel
is different from a failed measurement.

``invariant`` channels discard right-angle rotation and reflection, except for
ordinary rasterization error.  Most intrinsic ratios also discard translation
and uniform scale.  ``oriented`` channels deliberately retain canvas position,
orientation, and handedness.  ``quality`` channels describe observation quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import pi, sqrt
from typing import Iterable, Sequence

import cv2
import numpy as np

from .stroke_graph import thin, _graph


QUANTILES = (.0, .1, .25, .5, .75, .9, 1.)
STAT_NAMES = ("mean", "std", "min", "p10", "p25", "median", "p75", "p90", "max")


@dataclass(frozen=True)
class EncodedFeatures:
    values: np.ndarray
    valid: np.ndarray
    names: tuple[str, ...]
    groups: dict[str, tuple[int, ...]]
    diagnostics: tuple[dict, ...]
    encoder_id: str

    def __post_init__(self):
        if self.values.ndim != 2 or self.valid.shape != self.values.shape:
            raise ValueError("Feature values and validity must have equal N×D shapes")
        if self.values.shape[1] != len(self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("Feature names must uniquely identify every column")
        if len(self.diagnostics) != len(self.values) or not np.isfinite(self.values).all():
            raise ValueError("Feature values must be finite and diagnostics must match the batch")
        if self.valid.dtype != np.bool_:
            raise TypeError("Feature validity must be Boolean")

    @property
    def feature_names(self):
        return self.names

    def as_feature_batch(self, image_ids: Sequence[str], *, append_validity=True):
        """Adapt to the shared encoder contract without hiding missingness.

        Missing measurements remain zero with an explicit validity column.
        Scaling/feature selection must be fitted on support examples only.
        The adapter does not inspect any labels or other batch members.
        """
        from src.reasoning.features import FeatureBatch

        ids = tuple(str(item) for item in image_ids)
        if len(ids) != len(self.values):
            raise ValueError("Image IDs must match the number of encoded panels")
        indicators = self.valid.astype(np.float32)
        values = self.values.copy()
        names = self.names
        if append_validity:
            values = np.concatenate((values, indicators), axis=1)
            names += tuple("valid:"+name for name in self.names)
        family_indices = _feature_families(self.names)
        extra_views = {"measurements_only": self.values.copy(), "validity_indicators": indicators}
        view_names = {}
        for family, indices in family_indices.items():
            columns = np.asarray(indices, int)
            extra_views[family] = np.concatenate((self.values[:, columns], indicators[:, columns]), axis=1)
            selected_names = tuple(self.names[index] for index in indices)
            view_names[family] = list(selected_names+tuple("valid:"+name for name in selected_names))
        metadata = {"feature_names": list(names), "measurement_names": list(self.names),
                    "groups": {key: list(indices) for key, indices in self.groups.items()},
                    "feature_families": {key: list(indices) for key, indices in family_indices.items()},
                    "view_feature_names": view_names,
                    "automatic_feature_views": (["global"] if append_validity else [])+list(family_indices),
                    "diagnostic_feature_views": ["measurements_only", "validity_indicators"],
                    "missingness": "explicit validity indicators appended" if append_validity else "see validity_indicators view",
                    "normalization": "none fitted on this batch; use support-only scaling",
                    "diagnostics": list(self.diagnostics),
                    "invariance": "invariant channels: D4; oriented channels retain pose and handedness"}
        return FeatureBatch(image_ids=ids, global_features=values, patch_features=None,
                            encoder_id=self.encoder_id+("+validity" if append_validity else ""),
                            metadata=metadata,
                            extra_views=extra_views)


def _feature_families(names):
    """Measurement families, not classifiers or complete rule hypotheses."""
    groups = {name: [] for name in ("topology", "boundary", "morphology", "arrangement", "symmetry", "texture", "oriented")}
    for index, name in enumerate(names):
        scope, _, base = name.partition(".")
        if base.startswith(("component_count", "hole_", "euler_", "nesting_", "skeleton_endpoint_count", "skeleton_junction_count")) or base == "ink_present":
            groups["topology"].append(index)
        if base.startswith(("boundary.", "hu_abs_log_", "mass_angular.")):
            groups["boundary"].append(index)
        if ((base.startswith("component_") and not base.startswith(("component_count", "component_angular")))
                or base.startswith(("morphology_", "ink_width.", "skeleton_", "mass_covariance", "mass_anisotropy", "mass_radius."))
                or base in ("ink_fraction", "mass_rms_radius", "perimeter_per_sqrt_ink")):
            groups["morphology"].append(index)
        if base.startswith(("arrangement_", "component_angular.", "occupancy_", "center_")):
            groups["arrangement"].append(index)
        if base.startswith("symmetry_"):
            groups["symmetry"].append(index)
        if base.startswith(("gray_", "gradient_", "saturation_")):
            groups["texture"].append(index)
        if scope == "oriented":
            groups["oriented"].append(index)
    return {name: tuple(indices) for name, indices in groups.items() if indices}


class _Builder:
    def __init__(self):
        self.names, self.values, self.valid, self.groups = [], [], [], {}

    def add(self, name, value, group="invariant"):
        index = len(self.names)
        self.names.append(group+"."+name)
        good = value is not None and np.isfinite(value)
        self.values.append(float(value) if good else 0.)
        self.valid.append(bool(good))
        self.groups.setdefault(group, []).append(index)

    def stats(self, prefix, values, group="invariant"):
        array = np.asarray(values, float).reshape(-1)
        array = array[np.isfinite(array)]
        stats = ([float(array.mean()), float(array.std()), *np.quantile(array, QUANTILES)]
                 if len(array) else [None]*len(STAT_NAMES))
        for name, value in zip(STAT_NAMES, stats):
            self.add(prefix+"."+name, value, group)

    def arrays(self):
        return np.asarray(self.values, np.float32), np.asarray(self.valid, bool)


def _image(image, side):
    source = np.asarray(image)
    if source.ndim not in (2, 3) or min(source.shape[:2]) < 3:
        raise ValueError("Expected a grayscale, RGB, or RGBA panel at least 3×3 pixels")
    if source.ndim == 3 and source.shape[2] not in (1, 3, 4):
        raise ValueError("Expected one, three, or four image channels")
    if not np.issubdtype(source.dtype, np.number) and source.dtype != np.bool_:
        raise TypeError("Expected numeric image pixels")
    data = source.astype(np.float32)
    finite = np.isfinite(data)
    pixel_finite = np.all(finite, axis=2) if data.ndim == 3 else finite
    missing = float(1-np.mean(pixel_finite))
    finite_values = data[finite]
    if len(finite_values) and float(np.max(finite_values)) <= 1. and float(np.min(finite_values)) >= 0.:
        data *= 255
    data[~finite] = 255
    data = np.clip(data, 0, 255)
    if data.ndim == 2:
        rgb = np.repeat(data[:, :, None], 3, axis=2)
    elif data.shape[2] == 1:
        rgb = np.repeat(data, 3, axis=2)
    elif data.shape[2] == 4:
        alpha = data[:, :, 3:4]/255
        rgb = data[:, :, :3]*alpha+255*(1-alpha)
    else:
        rgb = data
    height, width = rgb.shape[:2]
    scale = side/max(height, width)
    resized = cv2.resize(rgb, (max(1, round(width*scale)), max(1, round(height*scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((side, side, 3), 255, np.uint8)
    y, x = (side-resized.shape[0])//2, (side-resized.shape[1])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = np.rint(resized).astype(np.uint8)
    diagnostics = {"status": "ok", "input_shape": list(source.shape), "analysis_side": side,
                   "nonfinite_fraction": missing, "warnings": []}
    if missing:
        diagnostics["warnings"].append("Nonfinite pixels were replaced with white; validity is restricted when damage exceeds 5%.")
    return canvas, diagnostics


def _covariance(points):
    if len(points) < 2:
        return None, None, None
    centered = points-points.mean(axis=0)
    matrix = centered.T@centered/len(centered)
    eigen = np.linalg.eigvalsh(matrix)
    trace = float(eigen.sum())
    if trace < 1e-9:
        return None, None, None
    return float(eigen[0]/trace), float(eigen[1]/trace), matrix/trace


def _canonical(mask, *, crop=False):
    """Choose one pixel-identical D4 representative, without repeated analysis.

    This removes direction bias in raster contour starts and thinning.  Only
    eight cheap array views are compared; the expensive measurements run once.
    """
    if crop and np.any(mask):
        yy, xx = np.where(mask)
        mask = mask[yy.min():yy.max()+1, xx.min():xx.max()+1]
    choices = [np.ascontiguousarray(view) for k in range(4)
               for view in (np.rot90(mask, k), np.rot90(mask, k)[:, ::-1])]
    return min(choices, key=lambda value: (value.shape, value.tobytes()))


def _harmonics(builder, prefix, vectors, orders=(1, 2, 3, 4, 6, 8, 12), weights=None):
    vectors = np.asarray(vectors, float).reshape(-1, 2)
    radii = np.linalg.norm(vectors, axis=1)
    good = radii > 1e-8
    if weights is not None:
        weight = np.asarray(weights, float)[good]
    else:
        weight = np.ones(np.count_nonzero(good))
    unit = (vectors[good, 0]+1j*vectors[good, 1])/radii[good]
    denominator = max(1e-12, float(weight.sum()))
    for order in orders:
        value = np.sum(unit**order*weight)/denominator if len(unit) else None
        builder.add(f"{prefix}.power_{order}", abs(value) if value is not None else None)
        builder.add(f"{prefix}.real_{order}", float(value.real) if value is not None else None, "oriented")
        builder.add(f"{prefix}.imag_{order}", float(value.imag) if value is not None else None, "oriented")


def _closed_resample(contour, count=128):
    points = np.asarray(contour, float).reshape(-1, 2)
    if len(points) < 3:
        return None
    points = np.vstack((points, points[0]))
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distances[-1] < 4:
        return None
    sample = np.linspace(0, distances[-1], count, endpoint=False)
    return np.column_stack([np.interp(sample, distances, points[:, axis]) for axis in (0, 1)])


def _boundary_descriptor(contour):
    points = _closed_resample(contour)
    if points is None:
        return None
    centered = points-points.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if scale < 1e-8:
        return None
    signal = (centered[:, 0]+1j*centered[:, 1])/scale
    spectrum = np.fft.fft(signal)/len(signal)
    descriptor = {f"fourier_{k}": float(np.hypot(abs(spectrum[k]), abs(spectrum[-k]))) for k in range(1, 13)}
    for step in (2, 4, 8):
        before = points-np.roll(points, step, axis=0)
        after = np.roll(points, -step, axis=0)-points
        angle = np.arctan2(before[:, 0]*after[:, 1]-before[:, 1]*after[:, 0], np.sum(before*after, axis=1))
        absolute = np.abs(angle)
        winding = np.sign(float(angle.sum())) or 1
        descriptor.update({f"turn_{step}.mean": float(absolute.mean()/pi),
                           f"turn_{step}.std": float(absolute.std()/pi),
                           f"turn_{step}.p90": float(np.quantile(absolute, .9)/pi),
                           f"turn_{step}.max": float(absolute.max()/pi),
                           f"turn_{step}.concave": float(np.mean(angle*winding < -.05))})
        if step == 4:
            powers = np.abs(np.fft.rfft(angle-angle.mean()))**2
            total = float(powers[1:].sum())
            for low, high in ((1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64)):
                descriptor[f"curvature_band_{low}_{high}"] = float(powers[low:high+1].sum()/total) if total > 1e-12 else 0.
    return descriptor


@lru_cache(maxsize=1)
def _boundary_keys():
    # The sample only defines names/order; it supplies no learned parameters.
    return tuple(_boundary_descriptor(np.array(((0, 0), (10, 0), (10, 10), (0, 10)))).keys())


def _iou(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b)/union if union else None


def _content_square(mask, size=96):
    yy, xx = np.where(mask)
    if not len(xx):
        return None
    crop = mask[yy.min():yy.max()+1, xx.min():xx.max()+1]
    length = max(crop.shape)+8
    canvas = np.zeros((length, length), np.uint8)
    y, x = (length-crop.shape[0])//2, (length-crop.shape[1])//2
    canvas[y:y+crop.shape[0], x:x+crop.shape[1]] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST) > 0


def _measure(rgb, diagnostics):
    builder = _Builder()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    side = len(gray)
    area = float(side*side)
    threshold, raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if float(gray.std()) < .5:
        raw = np.uint8(gray < 128)*255
    count, labels, stats, centers = cv2.connectedComponentsWithStats(raw, 8)
    ink_count = int(np.count_nonzero(raw))
    # Counts of tiny marks are retained separately; one-pixel threshold specks
    # do not dominate contour/skeleton measurements.
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 2]
    mask = np.uint8(np.isin(labels, keep))*255 if keep else np.zeros_like(raw)
    yy, xx = np.where(mask)
    points = np.column_stack((xx, yy)).astype(float)
    mass = len(points)
    centroid = points.mean(axis=0) if mass else np.array([side/2, side/2])
    centered = points-centroid
    rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))) if mass else 0.
    histogram = np.bincount(gray.reshape(-1), minlength=256)/area
    positive_hist = histogram[histogram > 0]
    entropy = float(-np.sum(positive_hist*np.log2(positive_hist))/8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    builder.add("input_valid", 1., "quality")
    builder.add("nonfinite_fraction", diagnostics["nonfinite_fraction"], "quality")
    builder.add("gray_entropy", entropy, "quality")
    builder.add("saturation_mean", float(hsv[:, :, 1].mean()/255), "quality")
    builder.add("saturation_p90", float(np.quantile(hsv[:, :, 1], .9)/255), "quality")
    builder.add("threshold", threshold/255, "quality")
    edge = np.r_[mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]]
    builder.add("border_ink_fraction", float(np.mean(edge > 0)), "quality")
    builder.add("speck_fraction", (ink_count-mass)/max(1, ink_count), "quality")
    builder.add("ink_present", float(mass > 0))
    builder.add("ink_fraction", mass/area)
    builder.add("component_count_raw", count-1)
    builder.add("component_count", len(keep))
    for fraction in (.0002, .001, .005, .02):
        builder.add(f"component_count_above_{fraction:g}", sum(stats[i, 4]/area >= fraction for i in keep))
    builder.add("gray_mean", float(gray.mean()/255))
    builder.add("gray_std", float(gray.std()/255))
    for q in (.1, .25, .5, .75, .9):
        builder.add(f"gray_quantile_{q:g}", float(np.quantile(gray, q)/255))
    builder.add("center_x", float(centroid[0]/side) if mass else None, "oriented")
    builder.add("center_y", float(centroid[1]/side) if mass else None, "oriented")
    minor, major, covariance = _covariance(points)
    builder.add("mass_covariance_minor", minor)
    builder.add("mass_covariance_major", major)
    builder.add("mass_anisotropy", major-minor if major is not None else None)
    builder.add("mass_covariance_xy", float(covariance[0, 1]) if covariance is not None else None, "oriented")
    builder.add("mass_covariance_xx", float(covariance[0, 0]) if covariance is not None else None, "oriented")
    builder.add("mass_rms_radius", rms/side if mass else None)
    builder.stats("mass_radius", np.linalg.norm(centered, axis=1)/rms if rms > 1e-8 else [])
    _harmonics(builder, "mass_angular", centered)
    canonical = _canonical(mask, crop=True)
    moments = cv2.moments(canonical, binaryImage=True)
    hu = cv2.HuMoments(moments).reshape(-1) if mass else None
    for index in range(7):
        value = float(hu[index]) if hu is not None else None
        builder.add(f"hu_abs_log_{index+1}", -np.log10(max(abs(value), 1e-30)) if value is not None else None)
    oriented_hu = cv2.HuMoments(cv2.moments(mask, binaryImage=True)).reshape(-1) if mass else None
    builder.add("hu_7_signed_log", float(np.sign(oriented_hu[6])*np.log1p(abs(oriented_hu[6])*1e12)) if oriented_hu is not None else None, "oriented")

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    depths = []
    if hierarchy is not None:
        for index in range(len(contours)):
            depth, parent = 0, int(hierarchy[0, index, 3])
            while parent >= 0:
                depth += 1
                parent = int(hierarchy[0, parent, 3])
            depths.append(depth)
    holes = [cv2.contourArea(c) for c, depth in zip(contours, depths) if depth % 2]
    builder.add("hole_count", len(holes))
    builder.add("euler_number", len(keep)-len(holes))
    builder.add("nesting_depth_max", max(depths, default=0))
    builder.add("hole_area_fraction", sum(holes)/area)
    builder.stats("hole_area", np.asarray(holes)/area)
    for fraction in (.0002, .001, .005, .02):
        builder.add(f"hole_count_above_{fraction:g}", sum(value/area >= fraction for value in holes))
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    builder.add("perimeter_per_sqrt_ink", perimeter/sqrt(mass) if mass else None)
    shape_values = {name: [] for name in ("area_fraction", "relative_area", "aspect", "extent", "solidity",
                                         "envelope_ink_ratio", "circularity", "concavity", "oriented_extent")}
    boundary_values = {key: [] for key in _boundary_keys()}
    for index in keep:
        x, y, width, height, ink_area = [int(v) for v in stats[index]]
        component = np.uint8(labels[y:y+height, x:x+width] == index)*255
        component = _canonical(component)
        comp_contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(comp_contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_mask = np.zeros_like(component)
        cv2.fillPoly(hull_mask, [hull], 255)
        hull_area = max(1, np.count_nonzero(hull_mask))
        envelope_area = max(1., cv2.contourArea(contour))
        line = cv2.arcLength(contour, True)
        _, oriented, _ = cv2.minAreaRect(contour)
        shape_values["area_fraction"].append(ink_area/area)
        shape_values["relative_area"].append(ink_area/max(1, mass))
        shape_values["aspect"].append(min(oriented)/max(1., max(oriented)))
        shape_values["extent"].append(ink_area/(width*height))
        shape_values["solidity"].append(min(1., ink_area/hull_area))
        shape_values["envelope_ink_ratio"].append(min(1., ink_area/max(1., envelope_area)))
        shape_values["circularity"].append(min(1., 4*pi*envelope_area/max(1., line**2)))
        shape_values["concavity"].append(max(0., 1-envelope_area/max(1., cv2.contourArea(hull))))
        shape_values["oriented_extent"].append(min(1., ink_area/max(1., oriented[0]*oriented[1])))
        description = _boundary_descriptor(contour)
        if description is not None:
            for key, value in description.items():
                boundary_values[key].append(value)
    for name, values in shape_values.items():
        builder.stats("component_"+name, values)
    for name, values in boundary_values.items():
        builder.add("boundary."+name+".mean", float(np.mean(values)) if values else None)
        builder.add("boundary."+name+".std", float(np.std(values)) if values else None)

    selected_centers = centers[keep] if keep else np.empty((0, 2))
    component_points = selected_centers-centroid
    _harmonics(builder, "component_angular", component_points)
    cm, cmaj, ccov = _covariance(selected_centers)
    builder.add("arrangement_covariance_minor", cm)
    builder.add("arrangement_covariance_major", cmaj)
    builder.add("arrangement_covariance_xy", float(ccov[0, 1]) if ccov is not None else None, "oriented")
    pair_distances, nearest, gaps, vectors = [], [], [], []
    if 1 < len(keep) <= 256:
        differences = selected_centers[:, None, :]-selected_centers[None, :, :]
        distance = np.linalg.norm(differences, axis=2)
        a, b = np.triu_indices(len(keep), 1)
        vectors = differences[a, b]
        pair_distances = distance[a, b]/(sqrt(2)*side)
        distance[np.diag_indices(len(keep))] = np.inf
        nearest = distance.min(axis=1)/(sqrt(2)*side)
        radii = np.sqrt(stats[keep, 4]/pi)
        gaps = (distance[a, b]-radii[a]-radii[b])/(sqrt(2)*side)
    elif len(keep) > 256:
        diagnostics["warnings"].append("Pairwise arrangement descriptors omitted above 256 ink components.")
    builder.stats("arrangement_pair_distance", pair_distances)
    builder.stats("arrangement_nearest_distance", nearest)
    builder.stats("arrangement_radius_gap", gaps)
    _harmonics(builder, "arrangement_pair_axis", vectors, orders=(2, 4, 6, 8))
    if len(keep) > 2:
        area_values = np.log(np.maximum(1, stats[keep, 4].astype(float)))
        centered_area = area_values-area_values.mean()
        positional = selected_centers-selected_centers.mean(axis=0)
        denom = float(np.linalg.norm(centered_area)*np.sqrt(np.sum(positional**2)))
        correlation = positional.T@centered_area/denom if denom > 1e-8 else None
    else:
        correlation = None
    builder.add("arrangement_area_gradient", float(np.linalg.norm(correlation)) if correlation is not None else None)
    builder.add("arrangement_area_gradient_x", float(correlation[0]) if correlation is not None else None, "oriented")
    builder.add("arrangement_area_gradient_y", float(correlation[1]) if correlation is not None else None, "oriented")

    for grid in (2, 4, 8):
        blocks = cv2.resize(mask.astype(np.float32)/255, (grid, grid), interpolation=cv2.INTER_AREA)
        for index, value in enumerate(blocks.reshape(-1)):
            builder.add(f"occupancy_{grid}.cell_{index}", float(value), "oriented")
        for index, value in enumerate(np.sort(blocks.reshape(-1))):
            builder.add(f"occupancy_{grid}.ordered_{index}", float(value))
    content = _content_square(mask)
    canonical_content = _content_square(canonical)
    if content is not None:
        reflections = [_iou(content, content[:, ::-1]), _iou(content, content[::-1]),
                       _iou(content, content.T), _iou(content, content[::-1, ::-1].T)]
        invariant_reflections = [_iou(canonical_content, canonical_content[:, ::-1]),
                                 _iou(canonical_content, canonical_content[::-1]),
                                 _iou(canonical_content, canonical_content.T),
                                 _iou(canonical_content, canonical_content[::-1, ::-1].T)]
        rotation90 = _iou(canonical_content, np.rot90(canonical_content))
        rotation180 = _iou(canonical_content, np.rot90(canonical_content, 2))
    else:
        reflections, invariant_reflections, rotation90, rotation180 = [None]*4, [None]*4, None, None
    builder.add("symmetry_rotation_90", rotation90)
    builder.add("symmetry_rotation_180", rotation180)
    for index, value in enumerate(sorted(invariant_reflections) if content is not None else invariant_reflections):
        builder.add(f"symmetry_reflection_ordered_{index}", value)
    for name, value in zip(("vertical", "horizontal", "diagonal", "anti_diagonal"), reflections):
        builder.add("symmetry_"+name, value, "oriented")

    # Canonicalization makes these discrete graph measurements D4 invariant
    # while requiring only one thinning pass, even on photographs.
    skeleton = thin(canonical) if mass else np.zeros_like(canonical)
    graph = _graph(skeleton)
    endpoint_pixels = [p for p, neighbors in graph.items() if len(neighbors) == 1]
    junction_pixels = [p for p, neighbors in graph.items() if len(neighbors) >= 3]
    junction_mask = np.zeros_like(canonical)
    for p in junction_pixels:
        junction_mask[p] = 255
    junction_count = cv2.connectedComponents(junction_mask, 8)[0]-1 if junction_pixels else 0
    graph_length = sum(distance for neighbors in graph.values() for _, distance in neighbors)/2
    builder.add("skeleton_endpoint_count", len(endpoint_pixels))
    builder.add("skeleton_junction_count", junction_count)
    builder.add("skeleton_pixel_fraction", np.count_nonzero(skeleton)/max(1, mass) if mass else None)
    builder.add("skeleton_length_per_sqrt_ink", graph_length/sqrt(mass) if mass else None)
    builder.add("skeleton_length_per_radius", graph_length/rms if rms > 1e-8 else None)
    endpoint_points = np.array([(x, y) for y, x in endpoint_pixels], float).reshape(-1, 2)
    junction_points = np.array([(x, y) for y, x in junction_pixels], float).reshape(-1, 2)
    canonical_y, canonical_x = np.where(canonical)
    canonical_center = np.array([canonical_x.mean(), canonical_y.mean()]) if mass else centroid
    builder.stats("skeleton_endpoint_radius", np.linalg.norm(endpoint_points-canonical_center, axis=1)/rms if rms > 1e-8 else [])
    builder.stats("skeleton_junction_radius", np.linalg.norm(junction_points-canonical_center, axis=1)/rms if rms > 1e-8 else [])
    builder.add("skeleton_endpoint_separation", float(np.linalg.norm(endpoint_points[0]-endpoint_points[1])/graph_length)
                if len(endpoint_pixels) == 2 and not junction_pixels and graph_length > 1e-8 else None)
    # Explicit outside background also bounds widths on a fully black panel;
    # OpenCV otherwise has no zero pixel and can return a huge sentinel radius.
    padded = np.pad(canonical, 1)
    distance_transform = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    builder.stats("ink_width", distance_transform[skeleton > 0]/sqrt(mass) if mass else [])
    for radius in (1, 2, 4, 8):
        kernel_y, kernel_x = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel = np.uint8(kernel_x**2+kernel_y**2 <= radius**2)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(mask, kernel)
        builder.add(f"morphology_open_{radius}_retained", np.count_nonzero(opened)/mass if mass else None)
        builder.add(f"morphology_close_{radius}_added", (np.count_nonzero(closed)-mass)/max(1, mass) if mass else None)
        builder.add(f"morphology_dilate_{radius}_components", cv2.connectedComponents(dilated, 8)[0]-1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    builder.stats("gradient_magnitude", magnitude.reshape(-1)/1020)
    gradients = np.column_stack((gx.reshape(-1), gy.reshape(-1)))
    _harmonics(builder, "gradient_axis", gradients, orders=(2, 4, 6, 8), weights=magnitude.reshape(-1))
    diagnostics.update(ink_components=len(keep), raw_components=count-1, ink_fraction=mass/area,
                       hole_count=len(holes), endpoint_count=len(endpoint_pixels), junction_count=junction_count,
                       gray_entropy=entropy, threshold=float(threshold))
    if len(edge) and np.mean(edge > 0) > .1:
        diagnostics["warnings"].append("Ink touches the panel border; topology may depend on the crop.")
    values, valid = builder.arrays()
    if diagnostics["nonfinite_fraction"] > .05:
        valid[np.asarray(builder.groups["invariant"]+builder.groups["oriented"], int)] = False
        values[~valid] = 0
        diagnostics["status"] = "damaged"
    return builder, values, valid


@lru_cache(maxsize=4)
def _schema(side):
    diagnostics = {"nonfinite_fraction": 0., "warnings": []}
    builder, _, _ = _measure(np.full((side, side, 3), 255, np.uint8), diagnostics)
    return tuple(builder.names), {key: tuple(value) for key, value in builder.groups.items()}


class StructuralEncoder:
    """Deterministic, CPU-only image encoder with no episode-dependent state."""

    def __init__(self, side=192, *, strict=False):
        if not isinstance(side, int) or side < 32 or side > 512:
            raise ValueError("Analysis side must be an integer between 32 and 512")
        self.side, self.strict = side, bool(strict)
        self.encoder_id = f"opencv-structural-v1-side{side}"
        self.names, schema_groups = _schema(side)
        self.groups = {key: tuple(indices) for key, indices in schema_groups.items()}

    @property
    def feature_names(self):
        return self.names

    def encode(self, images: Iterable[np.ndarray]) -> EncodedFeatures:
        if isinstance(images, np.ndarray) and images.ndim in (2, 3):
            images = [images]
        values, valid, diagnostics = [], [], []
        for image in images:
            try:
                rgb, report = _image(image, self.side)
                builder, row, available = _measure(rgb, report)
                if tuple(builder.names) != self.names:
                    raise RuntimeError("Structural feature schema changed within a batch")
            except (ValueError, TypeError, cv2.error) as error:
                if self.strict:
                    raise
                report = {"status": "invalid", "warnings": [str(error)]}
                row, available = np.zeros(len(self.names), np.float32), np.zeros(len(self.names), bool)
                index = self.names.index("quality.input_valid")
                available[index] = True
            values.append(row)
            valid.append(available)
            diagnostics.append(report)
        array = np.stack(values) if values else np.empty((0, len(self.names)), np.float32)
        available = np.stack(valid) if valid else np.empty((0, len(self.names)), bool)
        return EncodedFeatures(array, available, self.names, self.groups, tuple(diagnostics), self.encoder_id)


class StructuralFeatureEncoder:
    """Shared ``VisualEncoder`` adapter; fitting remains outside perception."""

    def __init__(self, side=192, *, strict=False, append_validity=True):
        self.measurements = StructuralEncoder(side=side, strict=strict)
        self.append_validity = bool(append_validity)
        self.encoder_id = self.measurements.encoder_id+("+validity" if self.append_validity else "")

    @property
    def feature_names(self):
        names = self.measurements.names
        return names+tuple("valid:"+name for name in names) if self.append_validity else names

    def encode(self, images: Iterable[np.ndarray], image_ids: Sequence[str] | None = None):
        encoded = self.measurements.encode(images)
        if image_ids is None:
            image_ids = tuple(f"panel-{index}" for index in range(len(encoded.values)))
        return encoded.as_feature_batch(image_ids, append_validity=self.append_validity)


def structural_descriptor(image, *, side=192):
    """Inspect one panel without exposing the feature learner or any labels."""
    encoded = StructuralEncoder(side=side).encode([image])
    return {"values": {name: float(value) if valid else None
                       for name, value, valid in zip(encoded.names, encoded.values[0], encoded.valid[0])},
            "diagnostics": encoded.diagnostics[0], "encoder_id": encoded.encoder_id}

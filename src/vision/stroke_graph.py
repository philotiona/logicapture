"""Geometry of arbitrary ink paths, independently of the primitive-shape CNN.

An arrow's direction is measured *along its own path*.  The traversal direction
is arbitrary; reversing it reverses every arrow sign and leaves the relation
between two arrows unchanged.  No problem labels are used by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
from math import sqrt
from typing import Any

import cv2
import numpy as np


@dataclass
class StrokeDiagnostics:
    """Inspectable measurements in pixel coordinates, not inferred rules."""

    path: Any
    skeleton: np.ndarray
    ordered_points: np.ndarray
    widths: np.ndarray
    candidates: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    orientation_evidence: list[dict] = field(default_factory=list)


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning; OpenCV-contrib is deliberately not required."""
    data = np.pad((mask > 0).astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for stage in (0, 1):
            p = data[1:-1, 1:-1]
            ns = (data[:-2, 1:-1], data[:-2, 2:], data[1:-1, 2:],
                  data[2:, 2:], data[2:, 1:-1], data[2:, :-2],
                  data[1:-1, :-2], data[:-2, :-2])
            count = sum(ns)
            transitions = sum((ns[i] == 0) & (ns[(i+1) % 8] == 1) for i in range(8))
            remove = (p == 1) & (count >= 2) & (count <= 6) & (transitions == 1)
            if stage == 0:
                remove &= (ns[0]*ns[2]*ns[4] == 0) & (ns[2]*ns[4]*ns[6] == 0)
            else:
                remove &= (ns[0]*ns[2]*ns[6] == 0) & (ns[0]*ns[4]*ns[6] == 0)
            if np.any(remove):
                p[remove] = 0
                changed = True
    return data[1:-1, 1:-1]*255


def _graph(skeleton):
    # Diagonal edges across an existing orthogonal connection introduce tiny
    # triangular graph cycles.  Suppressing them preserves the pixel topology.
    coords = {tuple(p) for p in np.argwhere(skeleton > 0)}
    graph = {}
    for y, x in coords:
        neighbors = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = (y+dy, x+dx)
            if q not in coords:
                continue
            if dy and dx and ((y, x+dx) in coords or (y+dy, x) in coords):
                continue
            neighbors.append((q, sqrt(2) if dy and dx else 1.))
        graph[(y, x)] = neighbors
    return graph


def _fill_microholes(binary, base_width):
    """Bridge tiny enclosed arrowhead/pen defects, preserving large path loops.

    Filled pixels are used for tracing only.  Arrow silhouette measurements
    always use the original ink, so this cannot invent a filled arrowhead.
    """
    filled = binary.copy()
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for contour, row in zip(contours, hierarchy[0]):
            if row[3] != -1 and cv2.contourArea(contour) <= max(9., 24*base_width**2):
                x, y, w, h = cv2.boundingRect(contour)
                if max(w, h) <= max(6., 8*base_width):
                    cv2.drawContours(filled, [contour], -1, 255, -1)
    return filled


def _collapse_solid_cycles(graph, mask, distance, base_width):
    """Remove raster graph loops lying inside solid ink, preserving ink holes.

    Pixel adjacency can leave a tiny cycle in a thinned solid arrowhead.  It
    is not a loop in the drawing: its enclosed pixels are also foreground.
    Real closed paths enclose background and are therefore never removed.
    """
    graph = {p: list(ns) for p, ns in graph.items()}
    limit = max(8, round(base_width*8))
    while True:
        parent, depth, tree_edges, seen = {}, {}, set(), set()
        for start in sorted(graph):
            if start in seen:
                continue
            parent[start], depth[start] = None, 0
            seen.add(start)
            stack = [start]
            while stack:
                p = stack.pop()
                for q, _ in graph[p]:
                    if q not in seen:
                        seen.add(q)
                        parent[q], depth[q] = p, depth[p]+1
                        tree_edges.add(frozenset((p, q)))
                        stack.append(q)
        cycles = []
        processed = set()
        for p, neighbors in graph.items():
            for q, _ in neighbors:
                edge = frozenset((p, q))
                if edge in tree_edges or edge in processed:
                    continue
                processed.add(edge)
                a, b, left, right = p, q, [p], [q]
                while a != b:
                    if depth[a] >= depth[b]:
                        a = parent[a]
                        left.append(a)
                    else:
                        b = parent[b]
                        right.append(b)
                cycle = left+list(reversed(right[:-1]))
                if len(cycle) < 3 or len(cycle) > limit:
                    continue
                contour = np.array([(x, y) for y, x in cycle], np.int32)
                x, y, w, h = cv2.boundingRect(contour)
                if max(w, h) > max(5, base_width*5):
                    continue
                interior = np.zeros_like(mask)
                cv2.fillPoly(interior, [contour], 255)
                pixels = mask[interior > 0]
                if len(pixels) and np.mean(pixels > 0) >= .95:
                    cycles.append(cycle)
        if not cycles:
            return graph
        cycle = min(cycles, key=len)
        edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        a, b = min(edges, key=lambda edge: (distance[edge[0]]+distance[edge[1]], edge))
        graph[a] = [(q, d) for q, d in graph[a] if q != b]
        graph[b] = [(q, d) for q, d in graph[b] if q != a]


def _prune_spurs(graph, limit, *, records=False):
    graph = {p: list(ns) for p, ns in graph.items()}
    pruned = []
    while True:
        eligible = []
        for endpoint in [p for p, ns in graph.items() if len(ns) == 1]:
            current, previous, chain, length = endpoint, None, [], 0.
            while True:
                chain.append(current)
                onward = [(q, d) for q, d in graph[current] if q != previous]
                if len(onward) != 1:
                    break
                nxt, distance = onward[0]
                length += distance
                previous, current = current, nxt
                if len(graph[current]) != 2:
                    break
            if len(graph[current]) > 2 and length <= limit:
                eligible.append({"chain": tuple(chain), "junction": current, "length": length})
        # A marker near a path endpoint has a short shaft continuation and a
        # short wing.  Removing both truncates the actual path at the marker.
        # Keep the arm that best continues the long trunk through the junction.
        preserve = set()
        for junction in {item["junction"] for item in eligible}:
            stubs = [item for item in eligible if item["junction"] == junction]
            if len(stubs) < 2:
                continue
            firsts = {item["chain"][-1] for item in stubs}
            trunk = [(q, d) for q, d in graph[junction] if q not in firsts]
            if len(trunk) != 1:
                continue
            previous, current = junction, trunk[0][0]
            walked = trunk[0][1]
            while walked < min(8., limit/3) and len(graph[current]) == 2:
                onward = [(q, d) for q, d in graph[current] if q != previous]
                new, length = onward[0]
                walked += length
                previous, current = current, new
            main_direction = np.array(current, float)-np.array(junction, float)
            main_direction /= max(1e-6, np.linalg.norm(main_direction))
            def alignment(item):
                arm = np.array(item["chain"][0], float)-np.array(junction, float)
                arm /= max(1e-6, np.linalg.norm(arm))
                return (float(np.dot(arm, main_direction)), -item["length"])
            kept = min(stubs, key=alignment)
            preserve.add(kept["chain"])
        eligible = [item for item in eligible if item["chain"] not in preserve]
        removable = {p for item in eligible for p in item["chain"]}
        pruned.extend(eligible)
        if not removable:
            return (graph, pruned) if records else graph
        graph = {p: [(q, d) for q, d in ns if q not in removable]
                 for p, ns in graph.items() if p not in removable}


def _distances(graph, start):
    distances, parents, queue = {start: 0.}, {}, [(0., start)]
    while queue:
        distance, p = heappop(queue)
        if distance != distances[p]:
            continue
        for q, edge in graph[p]:
            proposed = distance+edge
            if proposed < distances.get(q, float("inf")):
                distances[q], parents[q] = proposed, p
                heappush(queue, (proposed, q))
    return distances, parents


def _principal_path(graph):
    if not graph:
        return np.empty((0, 2)), "unknown", 0
    endpoints = sorted(p for p, ns in graph.items() if len(ns) == 1)
    junctions = [p for p, ns in graph.items() if len(ns) > 2]
    if not endpoints and not junctions:
        start = min(graph)
        order, previous, current = [], None, start
        while current not in order:
            order.append(current)
            nexts = sorted(q for q, _ in graph[current] if q != previous)
            if not nexts:
                break
            previous, current = current, nexts[0]
        return np.array([(x, y) for y, x in order], float), "closed", 0
    if len(endpoints) >= 2:
        longest, best = -1., None
        for start in endpoints:
            distances, parents = _distances(graph, start)
            for end in endpoints:
                distance = distances.get(end, -1)
                if distance > longest:
                    longest, best = distance, (start, end, parents)
        start, current, parents = best
        order = [current]
        while current != start:
            current = parents[current]
            order.append(current)
        order.reverse()
        return np.array([(x, y) for y, x in order], float), ("open" if not junctions else "branched"), len(junctions)
    # Junctions without endpoints can contain multiple loops; a single ordered
    # traversal would be invented evidence, so report the unresolved topology.
    return np.array([(x, y) for y, x in sorted(graph)], float), "branched", len(junctions)


def _graph_summary(graph, shape):
    """Collapse adjacent junction pixels; retain real arms and cycle counts."""
    h, w = shape
    endpoints = [p for p, ns in graph.items() if len(ns) == 1]
    junction_pixels = [p for p, ns in graph.items() if len(ns) > 2]
    junction_mask = np.zeros(shape, np.uint8)
    for p in junction_pixels:
        junction_mask[p] = 255
    n, labels, _, centers = cv2.connectedComponentsWithStats(junction_mask, 8)
    junctions = [tuple(point) for point in centers[1:]]
    edges = sum(len(ns) for ns in graph.values())//2
    # The caller passes one connected component.  Pruning preserves its core;
    # disconnected isolated skeleton pixels are counted explicitly for Euler.
    components, seen = 0, set()
    for start in graph:
        if start in seen:
            continue
        components += 1
        stack = [start]
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(q for q, _ in graph[p] if q not in seen)
    cycles = max(0, edges-len(graph)+components)
    visited, segments = set(), []
    terminals = sorted(p for p, ns in graph.items() if len(ns) != 2)
    for start in terminals:
        for nxt, _ in graph[start]:
            edge = frozenset((start, nxt))
            if edge in visited:
                continue
            order, previous, current = [start], start, nxt
            visited.add(edge)
            while True:
                order.append(current)
                if len(graph[current]) != 2:
                    break
                onward = [(q, d) for q, d in graph[current] if q != previous]
                if not onward:
                    break
                new = onward[0][0]
                edge = frozenset((current, new))
                if edge in visited:
                    break
                visited.add(edge)
                previous, current = current, new
            if len(order) >= 2:
                segments.append(tuple((x/w, y/h) for y, x in order))
    return {"endpoint_count": len(endpoints), "junction_count": len(junctions), "cycle_count": cycles,
            "endpoints": tuple((x/w, y/h) for y, x in endpoints),
            "junctions": tuple((x/w, y/h) for x, y in junctions),
            "segments": tuple(segments)}


def _resample(points, closed=False):
    if len(points) < 2:
        return points
    if closed:
        points = np.vstack([points, points[0]])
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    sample = np.arange(0, distances[-1], 1.)
    return np.column_stack([np.interp(sample, distances, points[:, axis]) for axis in (0, 1)])


def _smooth(values, radius, closed=False):
    radius = max(1, int(radius))
    padded = np.pad(values, ((radius, radius), (0, 0)) if values.ndim == 2 else (radius, radius),
                    mode="wrap" if closed else "edge")
    kernel = np.ones(2*radius+1)/(2*radius+1)
    if values.ndim == 1:
        return np.convolve(padded, kernel, "valid")
    return np.column_stack([np.convolve(padded[:, i], kernel, "valid") for i in range(values.shape[1])])


def _cross_sections(mask, points, width, closed, trace_mask=None):
    smooth_points = _smooth(points, max(2, width*1.5), closed)
    tangent = np.roll(smooth_points, -3, axis=0)-np.roll(smooth_points, 3, axis=0)
    if not closed:
        tangent[:3] = smooth_points[3:6]-smooth_points[:3]
        tangent[-3:] = smooth_points[-3:]-smooth_points[-6:-3]
    tangent /= np.maximum(1e-6, np.linalg.norm(tangent, axis=1))[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    offsets = np.arange(-max(8, width*5), max(8, width*5)+.25, .5)
    positions = points[:, None, :]+normal[:, None, :]*offsets[None, :, None]
    ink = cv2.remap(mask.astype(np.float32)/255, positions[:, :, 0].astype(np.float32),
                    positions[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # A normal can intersect a different part of a folded path.  Those ink
    # islands do not make this stroke wider.  Measure only the contiguous
    # cross-section containing its centerline.  Tiny marker holes are filled
    # for connectivity only; width still counts the original observed ink.
    connected_mask = mask if trace_mask is None else trace_mask
    connected = cv2.remap(connected_mask.astype(np.float32)/255,
                          positions[:, :, 0].astype(np.float32),
                          positions[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > .25
    # A hand-drawn marker wing can contain a sub-pen-width white crack while
    # staying attached longitudinally.  Bridge only that small sampling gap;
    # separated path legs remain separate normal components.
    connected = cv2.morphologyEx(connected.astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((1, max(2, round(width)+1)), np.uint8)) > 0
    midpoint = int(np.argmin(np.abs(offsets)))
    widths = np.zeros(len(points), float)
    side_widths = np.zeros((len(points), 2), float)
    for row, gate in enumerate(connected):
        center = midpoint
        if not gate[center]:
            hits = np.flatnonzero(gate)
            if not len(hits):
                continue
            center = int(hits[np.argmin(np.abs(hits-midpoint))])
            if abs(offsets[center]) > max(1., width):
                continue
        left, right = center, center
        while left > 0 and gate[left-1]:
            left -= 1
        while right+1 < len(gate) and gate[right+1]:
            right += 1
        widths[row] = ink[row, left:right+1].sum()*.5
        side_widths[row, 0] = ink[row, left:min(midpoint, right+1)].sum()*.5
        side_widths[row, 1] = ink[row, max(midpoint+1, left):right+1].sum()*.5
        if left <= midpoint <= right:
            side_widths[row] += ink[row, midpoint]*.25
    return widths, tangent, side_widths


def _analyze_orientation(mask: np.ndarray):
    """Measure one connected ink component in the panel coordinate system.

    ``mask`` is binary foreground (nonzero ink).  Arrow signs are relative to
    the returned ordered path, and never absolute screen angles.  Unresolved
    topology or arrow direction remains unknown.
    """
    from src.scene.strokes import ArrowMark, PathEvidence

    binary = (mask > 0).astype(np.uint8)*255
    height, width = binary.shape
    skeleton = thin(binary)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radius_samples = distance[skeleton > 0]
    base_width = max(1., float(np.percentile(radius_samples, 30))*2-1) if len(radius_samples) else 1.
    trace_mask = _fill_microholes(binary, base_width)
    skeleton = thin(trace_mask)
    raw_graph = _collapse_solid_cycles(_graph(skeleton), trace_mask, distance, base_width)
    graph, pruned = _prune_spurs(raw_graph, max(6, base_width*8.), records=True)
    ordered, topology, junctions = _principal_path(graph)
    ordered = _resample(ordered, topology == "closed")
    warnings = []
    if topology == "branched":
        warnings.append("A connected branching stroke has no unique path traversal.")
    if len(ordered) < 2:
        result = StrokeDiagnostics(None, skeleton, ordered, np.zeros(len(ordered)), [],
                                   ["Too little centerline geometry to measure a path."])
        return result
    widths, candidates, arrows = np.zeros(len(ordered)), [], []
    if len(ordered) >= 12 and topology in ("open", "closed"):
        widths, tangent, sides = _cross_sections(binary, ordered, base_width, topology == "closed", trace_mask)
        candidates = _arrow_candidates(binary, ordered, widths, tangent, base_width, topology == "closed", sides)
        for c in candidates:
            arrows.append(ArrowMark(arc_position=c["index"]/max(1, len(ordered)-1), direction=c["direction"],
                                    confidence=c["confidence"],
                                    tip=tuple(c["tip"]/np.array([width, height])),
                                    tail=tuple(c["tail"]/np.array([width, height]))))
    unexplained = []
    for spur in pruned:
        if spur["length"] <= max(2., base_width*1.25):
            continue  # A pen-width corner spur is a skeleton discretization.
        y, x = spur["junction"]
        supported = any(c["head_aspect"] >= .9 and
                        (c["bilateral_flare"] or (c["direction"] != 0 and c["confidence"] >= .65)) and
                        np.linalg.norm(ordered[c["index"]]-np.array([x, y])) <= max(5., base_width*5)
                        for c in candidates)
        if not supported:
            unexplained.append(spur)
    if unexplained:
        # Short branches cannot be silently erased as if every small arm were
        # an arrow wing.  Restore them when no reliable marker explains them.
        retained = set(graph)
        for spur in unexplained:
            retained.update(spur["chain"])
            retained.add(spur["junction"])
        graph = {p: [(q, d) for q, d in raw_graph[p] if q in retained] for p in retained}
        topology = "branched"
        warnings.append("An unexplained short branch was retained instead of being treated as an arrow wing.")
    points = tuple(tuple(p/np.array([width, height])) for p in ordered[::max(1, len(ordered)//400)])
    path = PathEvidence(points=points, topology=topology, arrows=tuple(arrows),
                        confidence=.9 if topology in ("open", "closed") else .35,
                        topology_confidence=.9 if topology != "unknown" else .35,
                        width=base_width/max(width, height), **_graph_summary(graph, binary.shape))
    result = StrokeDiagnostics(path, skeleton, ordered, widths, candidates, warnings)
    return result


def _untransform_points(points, shape, turns, flip, *, normalized=False):
    """Undo one D4 image transform without changing the path traversal order."""
    if not len(points):
        return np.empty((0, 2))
    h, w = shape
    rh, rw = (w, h) if turns % 2 else (h, w)
    array = np.array(points, float).reshape(-1, 2).copy()
    if normalized:
        array *= np.array([rw, rh])
    x, y = array[:, 0].copy(), array[:, 1].copy()
    if flip:
        x = rw-1-x
    if turns == 0:
        mapped = np.column_stack([x, y])
    elif turns == 1:
        mapped = np.column_stack([w-1-y, x])
    elif turns == 2:
        mapped = np.column_stack([w-1-x, h-1-y])
    else:
        mapped = np.column_stack([y, h-1-x])
    if normalized:
        mapped /= np.array([w, h])
    return mapped


def _untransform_result(result, shape, turns, flip):
    if result.path is None:
        return result
    def normalized_tuple(points):
        return tuple(tuple(float(v) for v in p) for p in
                     _untransform_points(points, shape, turns, flip, normalized=True))
    def optional_point(point):
        return normalized_tuple((point,))[0] if point is not None else None
    arrows = tuple(replace(a, tip=optional_point(a.tip), tail=optional_point(a.tail)) for a in result.path.arrows)
    path = replace(result.path, points=normalized_tuple(result.path.points), arrows=arrows,
                   endpoints=normalized_tuple(result.path.endpoints),
                   junctions=normalized_tuple(result.path.junctions),
                   segments=tuple(normalized_tuple(segment) for segment in result.path.segments))
    skeleton = result.skeleton[:, ::-1] if flip else result.skeleton
    skeleton = np.rot90(skeleton, -turns).copy()
    candidates = [dict(c, tip=_untransform_points((c["tip"],), shape, turns, flip)[0],
                       tail=_untransform_points((c["tail"],), shape, turns, flip)[0])
                  for c in result.candidates]
    return StrokeDiagnostics(path, skeleton,
                             _untransform_points(result.ordered_points, shape, turns, flip),
                             result.widths, candidates, list(result.warnings))


def analyze_stroke(mask: np.ndarray, *, diagnostics=False):
    """Measure one connected ink component, preserving direction uncertainty.

    The eight pixel-preserving quarter-turn/reflection orientations form a
    geometric measurement ensemble.  Thinning and normal sampling have small
    grid biases; an accepted direction relation needs agreement from at least
    two orientations and no confident contradictory interpretation.
    Evidence is transformed back to the caller's pixel coordinates.  This
    mechanism has no access to Bongard groups, rule labels, or drawing names.

    Return ``None`` when the ink has no measurable centerline.  A branched path
    retains its graph measurements while its arrow relation stays unknown.
    """
    binary = np.asarray(mask)
    if binary.ndim != 2 or min(binary.shape) < 3:
        raise ValueError("Expected a two-dimensional ink mask at least 3×3 pixels")
    binary = (binary > 0).astype(np.uint8)*255
    orientations = []
    for turns in range(4):
        rotated = np.rot90(binary, turns)
        for flip in (False, True):
            transformed = rotated[:, ::-1].copy() if flip else rotated.copy()
            result = _analyze_orientation(transformed)
            if result.path is not None:
                orientations.append((result, turns, flip, transformed.tobytes()))
    if not orientations:
        empty = _analyze_orientation(binary)
        return empty if diagnostics else None
    relations = [r.path.arrow_relation for r, _, _, _ in orientations]
    resolved = {relation for relation in relations if relation != "unknown"}
    topology_votes = {topology: sum(r.path.topology == topology for r, _, _, _ in orientations)
                      for topology in ("open", "closed", "branched", "unknown")}
    selected_topology = max(topology_votes, key=topology_votes.get)
    def quality(item):
        r, _, _, key = item
        p = r.path
        # This is an evidence quality score, not a calibrated probability.
        arrows = p.arrows
        confident = p.arrow_relation != "unknown"
        arrow_quality = min((float(a.confidence) for a in arrows), default=0.)
        return (p.topology == selected_topology, confident, arrow_quality, -len(arrows), key)
    selected, turns, flip, _ = max(orientations, key=quality)
    result = _untransform_result(selected, binary.shape, turns, flip)
    if topology_votes[selected_topology] <= len(orientations)/2:
        result.warnings.append("Grid orientations do not agree on a unique stroke topology.")
        result.path = replace(result.path, topology="unknown", confidence=min(.6, result.path.confidence),
                              topology_confidence=.6, endpoint_count=None, junction_count=None, cycle_count=None)
    supported = len([r for r in relations if r != "unknown"])
    contradictory = len(resolved) > 1
    if contradictory or (resolved and supported < 2):
        reason = "Grid orientations disagree about arrow direction." if contradictory else "Too few grid orientations support the arrow relation."
        result.warnings.append(reason)
        result.path = replace(result.path, arrows=tuple(replace(a, direction=0, confidence=min(.6, a.confidence))
                                                        for a in result.path.arrows))
    result.orientation_evidence = [{"turns": k, "reflected": reflected,
                                    "topology": r.path.topology, "arrow_relation": r.path.arrow_relation,
                                    "arrow_count": len(r.path.arrows),
                                    "minimum_arrow_confidence": min((float(a.confidence) for a in r.path.arrows), default=0.)}
                                   for r, k, reflected, _ in orientations]
    return result if diagnostics else result.path


def _arrow_candidates(mask, points, widths, tangent, base_width, closed, sides):
    # Localized widening is only a candidate, not itself an arrow.  Direction
    # needs the asymmetric broad-base / pointed-tip silhouette along the path.
    baseline = max(base_width, float(np.percentile(widths, 35)))
    profile = _smooth(widths, 1, closed)
    excess = np.maximum(0, profile-baseline)
    active = excess > max(.6, baseline*.35)
    intervals = []
    current = []
    for i, on in enumerate(active):
        if on:
            current.append(i)
        elif current:
            intervals.append(current)
            current = []
    if current:
        intervals.append(current)
    if closed and len(intervals) > 1 and active[0] and active[-1]:
        intervals[0] = intervals[-1]+intervals[0]
        intervals.pop()
    candidates = []
    for interval in intervals:
        if len(interval) < 2:
            continue
        indices = np.array(interval)
        peak = indices[np.argmax(excess[indices])]
        neighborhood = np.arange(int(peak)-max(8, round(baseline*10)),
                                 int(peak)+max(8, round(baseline*10))+1)
        if not closed:
            neighborhood = neighborhood[(neighborhood >= 0) & (neighborhood < len(profile))]
        local_baseline = max(baseline, float(np.percentile(profile[neighborhood % len(profile)], 35)))
        local_excess = np.maximum(0, profile-local_baseline)
        strength = float(local_excess[peak])
        if strength < max(1., local_baseline*.6):
            continue
        # Recover near-baseline wings outside the detection interval.
        a, b = int(peak), int(peak)
        limit = max(5, round(baseline*9))
        for sign in (-1, 1):
            j = peak
            for step in range(1, limit+1):
                k = (peak+sign*step) % len(profile)
                if not closed and not 0 <= peak+sign*step < len(profile):
                    break
                if local_excess[k] < max(.15, strength*.08):
                    break
                j = peak+sign*step
            if sign == -1:
                a = j
            else:
                b = j
        local = np.arange(a, b+1)
        weights = local_excess[local % len(excess)]
        top = weights >= strength*.95
        peak_position = float(np.mean(local[top])) if np.any(top) else float(peak)
        # Center a flat maximum instead of selecting its first raster pixel.
        # Otherwise a perfectly symmetric ellipse acquires an artificial
        # taper sign simply because np.argmax returns the leftmost maximum.
        centered = local-peak_position
        mean_position = float(np.sum(weights*local)/max(1e-6, weights.sum()))
        variance = float(np.sum(weights*(local-mean_position)**2)/max(1e-6, weights.sum()))
        third_moment = float(np.sum(weights*(local-mean_position)**3)/max(1e-6, weights.sum()))
        moment_skewness = third_moment/max(1., variance**1.5)
        # A sharp bend increases a normal cross-section even for a constant
        # width pen.  It is not an attached arrowhead.  Compare headings on
        # the two sides of the widening before interpreting its taper.
        before = tangent[a % len(tangent)]
        after = tangent[b % len(tangent)]
        turn = float(np.degrees(np.arccos(np.clip(np.dot(before, after), -1, 1))))
        if turn > 65:
            continue
        # Triangular heads have a steep rear edge and a longer taper to the
        # tip; this moment is positive when the tail of that taper is forward.
        skew = float(np.sum(weights*centered)/max(1e-6, weights.sum()))
        direction = 1 if skew > 0 else -1
        asymmetry = abs(skew)/max(1., (b-a)/2)
        confidence = min(.96, .5+asymmetry*1.5)
        method = "peak-relative taper"
        # A broad rasterized plateau moves the argmax by a pixel or two.  The
        # third central moment does not depend on that maximum's location.
        # A triangular taper's ideal skewness is 2*sqrt(2)/5; compare against
        # that geometric reference, rather than treating a shifted peak as
        # evidence of reversal.  Opposing strong cues remain unresolved.
        ideal_triangle_skew = 2*sqrt(2)/5
        moment_confidence = min(.96, .5+.6*abs(moment_skewness)/ideal_triangle_skew)
        if abs(moment_skewness) > 2*ideal_triangle_skew:
            # A very heavy residual tail is unlike a triangular head.  It
            # commonly comes from a second ink feature or a tracing offset;
            # magnitude alone must not turn that mismatch into certainty.
            moment_confidence = .5
        moment_direction = 1 if moment_skewness > 0 else -1
        if confidence < .65 and moment_confidence >= .65:
            direction, confidence = moment_direction, moment_confidence
            method = "central-moment taper"
        elif confidence >= .65 and moment_confidence >= .65 and direction != moment_direction:
            direction, confidence = 0, .5
            method = "contradictory taper cues"
        if abs(moment_skewness) < ideal_triangle_skew*.075:
            direction, confidence = 0, min(.6, confidence)
            method = "symmetric widening"
        if confidence < .65:
            direction = 0
        end = b if direction > 0 else a
        rear = a if direction > 0 else b
        side_baselines = np.percentile(sides[neighborhood % len(sides)], 35, axis=0)
        bilateral = np.max(np.minimum(*(sides[indices]-side_baselines).T)) >= max(.25, base_width*.15)
        candidates.append({"index": int(peak), "direction": direction, "confidence": confidence,
                           "tip": points[end % len(points)], "tail": points[rear % len(points)],
                           "width": float(profile[peak]), "base_width": local_baseline,
                           "interval": (a, b), "skew": skew, "asymmetry": asymmetry})
        candidates[-1]["heading_change"] = turn
        candidates[-1]["moment_skewness"] = moment_skewness
        candidates[-1]["direction_method"] = method
        candidates[-1]["peak_position"] = peak_position
        candidates[-1]["head_aspect"] = (b-a)/max(1., strength)
        candidates[-1]["bilateral_flare"] = bool(bilateral)
    # One jagged head may contain multiple above-baseline runs.  Merge their
    # overlapping arc supports, rather than counting each run as an arrow.
    groups = []
    for candidate in candidates:
        support = set(i % len(points) for i in range(candidate["interval"][0], candidate["interval"][1]+1))
        matches = [group for group in groups if len(support & group["support"]) >= .5*min(len(support), len(group["support"]))]
        if not matches:
            groups.append({"support": support, "items": [candidate]})
        else:
            selected = matches[0]
            selected["support"].update(support)
            selected["items"].append(candidate)
            for other in matches[1:]:
                selected["support"].update(other["support"])
                selected["items"].extend(other["items"])
                groups.remove(other)
    merged = []
    for group in groups:
        items = group["items"]
        best = dict(max(items, key=lambda c: (c["confidence"], c["width"])))
        signs = {c["direction"] for c in items if c["confidence"] >= .65 and c["direction"] != 0}
        if len(signs) > 1:
            best.update(direction=0, confidence=.5, direction_method="overlapping contradictory markers")
        best["merged_candidates"] = len(items)
        merged.append(best)
    return merged

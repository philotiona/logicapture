"""Geometry in normalized panel units, independent of the image library."""
from dataclasses import dataclass
from itertools import combinations
from math import hypot

from .objects import Scene


@dataclass(frozen=True)
class GeometryTolerance:
    position_margin: float = 0.05
    near_distance: float = 0.30
    touching_margin: float = 0.012
    size_ratio: float = 1.20
    alignment_margin: float = 0.025

    def __post_init__(self):
        if min(self.position_margin, self.near_distance, self.touching_margin,
               self.alignment_margin) < 0 or self.size_ratio < 1:
            raise ValueError("Geometric tolerances must be nonnegative; size ratio >= 1")


def point_inside(point, polygon):
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y):
            cross = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < cross:
                inside = not inside
    return inside


def _convex_gap(a, b):
    """Separating axis gap; positive when convex outlines are disjoint."""
    gap = -float("inf")
    for poly in (a, b):
        for p, q in zip(poly, poly[1:] + poly[:1]):
            dx, dy = q[0] - p[0], q[1] - p[1]
            length = hypot(dx, dy)
            if length == 0:
                continue
            nx, ny = -dy / length, dx / length
            pa = [x * nx + y * ny for x, y in a]
            pb = [x * nx + y * ny for x, y in b]
            gap = max(gap, min(pb) - max(pa), min(pa) - max(pb))
    return gap


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _point_segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/max(dx*dx+dy*dy, 1e-18)))
    return hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


def _polygon_contact(a, b):
    """Boundary distance and proper crossings for arbitrary simple contours.

    The old separating-axis test assumed convexity and invented overlap across
    empty concave notches. Keep the actual boundary instead of its convex hull.
    """
    distance, crosses = float("inf"), False
    for p, q in zip(a, a[1:]+a[:1]):
        for r, s in zip(b, b[1:]+b[:1]):
            proper = _cross(p, q, r)*_cross(p, q, s) < -1e-18 and _cross(r, s, p)*_cross(r, s, q) < -1e-18
            if proper:
                crosses, distance = True, 0.
            else:
                distance = min(distance, _point_segment_distance(p, r, s), _point_segment_distance(q, r, s),
                               _point_segment_distance(r, p, q), _point_segment_distance(s, p, q))
    return distance, crosses


def derive_relations(scene: Scene, tolerance=GeometryTolerance()) -> Scene:
    result = set()
    vertices = {o.id: o.vertices() for o in scene.objects}
    contacts = {}
    def contact(a, b):
        key = tuple(sorted((a.id, b.id)))
        if key not in contacts:
            contacts[key] = _polygon_contact(vertices[a.id], vertices[b.id])
        return contacts[key]
    def contains(a, b):
        va, vb = vertices[a.id], vertices[b.id]
        if not a.area < b.area or not all(point_inside(p, vb) for p in va):
            return False
        if contact(a, b)[1]:
            return False
        return all(point_inside(((p[0]+q[0])/2, (p[1]+q[1])/2), vb)
                   for p, q in zip(va, va[1:]+va[:1]))
    for a in scene.objects:
        for b in scene.objects:
            if a.id == b.id:
                continue
            def add(name, condition):
                if condition:
                    result.add((name, a.id, b.id))
            add("left_of", a.center_x < b.center_x - tolerance.position_margin)
            add("right_of", a.center_x > b.center_x + tolerance.position_margin)
            add("above", a.center_y < b.center_y - tolerance.position_margin)
            add("below", a.center_y > b.center_y + tolerance.position_margin)
            add("near", hypot(a.center_x - b.center_x, a.center_y - b.center_y) < tolerance.near_distance)
            add("larger_than", a.area > b.area * tolerance.size_ratio)
            add("smaller_than", b.area > a.area * tolerance.size_ratio)
            add("same_size", max(a.area, b.area) <= min(a.area, b.area) * tolerance.size_ratio)
            add("same_shape", a.shape == b.shape and a.shape != "unknown")
            add("same_color", a.color == b.color)
            if a.kind != "region" or b.kind != "region":
                # An open path is not a filled polygon. Undefined topological
                # region relations are left for Rule.assess to abstain on.
                continue
            if len(vertices[a.id]) < 3 or len(vertices[b.id]) < 3:
                continue
            inside = contains(a, b)
            add("inside", inside)
            if a.id < b.id:
                va, vb = vertices[a.id], vertices[b.id]
                # Bounding boxes avoid expensive polygon checks for distant objects.
                bbox_gap = max(min(x for x, y in vb) - max(x for x, y in va),
                               min(x for x, y in va) - max(x for x, y in vb),
                               min(y for x, y in vb) - max(y for x, y in va),
                               min(y for x, y in va) - max(y for x, y in vb))
                if bbox_gap <= tolerance.touching_margin:
                    gap, crosses = contact(a, b)
                    contained = inside or contains(b, a)
                    overlap = (crosses or any(point_inside(p, vb) for p in va) or any(point_inside(p, va) for p in vb)) and not contained
                    for name, condition in (("touching", gap <= tolerance.touching_margin and not overlap),
                                            ("overlapping", overlap)):
                        if condition:
                            result.update({(name, a.id, b.id), (name, b.id, a.id)})
    for a, b, c in combinations(scene.objects, 3):
        distance = abs((b.center_x-a.center_x)*(c.center_y-a.center_y) -
                       (b.center_y-a.center_y)*(c.center_x-a.center_x)) / max(
                           hypot(b.center_x-a.center_x, b.center_y-a.center_y), 1e-9)
        if distance <= tolerance.alignment_margin:
            result.add(("aligned", *sorted((a.id, b.id, c.id))))
    scene.relations = result
    return scene

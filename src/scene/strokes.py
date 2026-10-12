"""Geometry-first evidence for arbitrary strokes, independent of primitive shapes.

Arrow directions refer to an ordered traversal of a centerline.  Comparing those
signs expresses following/opposing the path, even when the image tangents point
in different directions.  Reversing the traversal flips every sign together.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot

EVIDENCE_THRESHOLD = .65
TOPOLOGIES = ("open", "closed", "branched", "unknown")


def _distribution(distribution, labels):
    if distribution is not None:
        if not distribution or any(k not in labels or not 0 <= v <= 1 for k, v in distribution.items()):
            raise ValueError("Invalid evidence probability distribution")
        if abs(sum(distribution.values())-1) > 1e-4:
            raise ValueError("Evidence probabilities must sum to one")


@dataclass(frozen=True)
class ArrowMark:
    arc_position: float
    direction: int = 0
    confidence: float = 0.
    tip: tuple[float, float] | None = None
    tail: tuple[float, float] | None = None
    probabilities: dict[str, float] | None = None
    presence_confidence: float | None = None

    def __post_init__(self):
        if not 0 <= self.arc_position <= 1 or self.direction not in (-1, 0, 1) or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid arrow evidence")
        for point in (self.tip, self.tail):
            if point is not None and (len(point) != 2 or any(not 0 <= x <= 1 for x in point)):
                raise ValueError("Arrow coordinates must be normalized")
        _distribution(self.probabilities, ("forward", "backward", "unknown"))
        if self.presence_confidence is not None and not 0<=self.presence_confidence<=1:
            raise ValueError("Invalid marker presence confidence")

    @property
    def reliable(self):
        return self.direction != 0 and self.confidence >= EVIDENCE_THRESHOLD

    @property
    def detected(self):
        strength=self.confidence if self.presence_confidence is None else self.presence_confidence
        return strength>=EVIDENCE_THRESHOLD

    def reversed(self):
        probabilities = None if self.probabilities is None else {
            {"forward": "backward", "backward": "forward", "unknown": "unknown"}[key]: value
            for key, value in self.probabilities.items()}
        return replace(self, arc_position=1-self.arc_position, direction=-self.direction, probabilities=probabilities)


@dataclass(frozen=True)
class PathEvidence:
    points: tuple[tuple[float, float], ...]
    topology: str = "unknown"
    arrows: tuple[ArrowMark, ...] = ()
    confidence: float = 1.
    topology_probabilities: dict[str, float] | None = None
    width: float = .01
    topology_confidence: float | None = None
    endpoint_count: int | None = None
    junction_count: int | None = None
    cycle_count: int | None = None
    endpoints: tuple[tuple[float, float], ...] = ()
    junctions: tuple[tuple[float, float], ...] = ()
    segments: tuple[tuple[tuple[float, float], ...], ...] = ()

    def __post_init__(self):
        if len(self.points) < 2 or any(len(p) != 2 or any(not 0 <= x <= 1 for x in p) for p in self.points):
            raise ValueError("A centerline needs at least two normalized points")
        if self.topology not in TOPOLOGIES or not 0 <= self.confidence <= 1 or not 0 < self.width <= 1:
            raise ValueError("Invalid path evidence")
        if any(not isinstance(a, ArrowMark) for a in self.arrows):
            raise TypeError("Path arrows must be ArrowMark evidence")
        _distribution(self.topology_probabilities, TOPOLOGIES)
        if self.topology_confidence is not None and not 0 <= self.topology_confidence <= 1:
            raise ValueError("Invalid topology confidence")
        for value in (self.endpoint_count,self.junction_count,self.cycle_count):
            if value is not None and (not isinstance(value,int) or value < 0):
                raise ValueError("Graph counts must be nonnegative integers")
        points = (*self.endpoints,*self.junctions,*[point for segment in self.segments for point in segment])
        if any(len(p)!=2 or any(not 0<=x<=1 for x in p) for p in points):
            raise ValueError("Graph coordinates must be normalized")

    @property
    def structural_confidence(self):
        return self.confidence if self.topology_confidence is None else self.topology_confidence

    def graph_count(self,name):
        value = getattr(self,name+"_count")
        if value is not None: return value
        if self.topology in ("open","closed"):
            return {"endpoint":2 if self.topology=="open" else 0,"junction":0,"cycle":1 if self.topology=="closed" else 0}[name]
        return None

    @property
    def arrow_relation(self):
        """Consensus/disagreement along a path; incomplete evidence stays unknown."""
        if self.confidence < EVIDENCE_THRESHOLD or self.topology in ("branched", "unknown"):
            return "unknown"
        reliable = [a.direction for a in self.arrows if a.reliable]
        if -1 in reliable and 1 in reliable:
            return "opposed"
        if len(reliable) >= 2 and len(reliable) == len(self.arrows):
            return "same"
        return "unknown"

    @property
    def direction_confidence(self):
        if self.arrow_relation == "unknown": return 0.
        if self.arrow_relation == "same": return min(self.confidence, *(a.confidence for a in self.arrows))
        forward = max(a.confidence for a in self.arrows if a.reliable and a.direction == 1)
        backward = max(a.confidence for a in self.arrows if a.reliable and a.direction == -1)
        return min(self.confidence, forward, backward)

    @property
    def length(self):
        return sum(hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(self.points, self.points[1:]))

    @property
    def geometry(self):
        """A rotation-invariant broad cue, never a polygon class."""
        if self.confidence < EVIDENCE_THRESHOLD or self.topology == "branched": return "unknown"
        if self.topology == "closed": return "curved"
        first,last = self.points[0],self.points[-1]
        dx,dy = last[0]-first[0],last[1]-first[1]
        distance = hypot(dx,dy)
        if distance <= 1e-8: return "unknown"
        # Pixel staircases inflate arc length for diagonal lines. Measure the
        # transverse residual instead, normalized by span and independent of
        # rotation, reflection, traversal or centerline sampling density.
        residuals = [abs(dx*(p[1]-first[1])-dy*(p[0]-first[0]))/distance for p in self.points]
        rms = (sum(r*r for r in residuals)/len(residuals))**.5
        return "straight" if rms/distance <= .018 else "curved"

    def point_tangent(self, position):
        """Interpolate by arc length, with a stable tangent for drawing arrows."""
        distances = [hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(self.points, self.points[1:])]
        remaining = max(0., min(1., position))*sum(distances)
        for index, length in enumerate(distances):
            if remaining <= length or index == len(distances)-1:
                a, b = self.points[index:index+2]
                t = remaining/length if length else 0.
                return ((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])),
                        ((b[0]-a[0])/max(length, 1e-9), (b[1]-a[1])/max(length, 1e-9)))
            remaining -= length

    def reversed(self):
        return replace(self, points=tuple(reversed(self.points)), arrows=tuple(a.reversed() for a in reversed(self.arrows)))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["points"] = tuple(tuple(p) for p in data["points"])
        arrows = []
        for arrow in data.get("arrows", ()):
            if isinstance(arrow, ArrowMark): arrows.append(arrow); continue
            arrow = dict(arrow)
            for key in ("tip", "tail"):
                if arrow.get(key) is not None: arrow[key] = tuple(arrow[key])
            arrows.append(ArrowMark(**arrow))
        data["arrows"] = tuple(arrows)
        for key in ("endpoints","junctions"):
            data[key] = tuple(tuple(point) for point in data.get(key,()))
        data["segments"] = tuple(tuple(tuple(point) for point in segment) for segment in data.get("segments",()))
        return cls(**data)

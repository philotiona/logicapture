from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import cos, sin, pi, sqrt
from .strokes import PathEvidence

SHAPES = ("circle", "triangle", "quadrilateral", "polygon", "unknown")
COLORS = ("black", "red", "blue", "green")


@dataclass(frozen=True)
class VisualObject:
    id: int
    shape: str
    filled: bool
    color: str = "black"
    center_x: float = 0.5
    center_y: float = 0.5
    area: float = 0.04
    angle: float = 0.0
    contour: tuple[tuple[float, float], ...] = ()
    confidence: float = 1.0
    shape_probabilities: dict[str, float] | None = None
    fill_probabilities: dict[str, float] | None = None
    attribute_source: str = "symbolic"
    kind: str = "region"
    path: PathEvidence | None = None

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"Unsupported shape: {self.shape}")
        if self.kind not in ("region", "path", "unknown"):
            raise ValueError("Unsupported visual entity kind")
        if self.path is not None and not isinstance(self.path, PathEvidence):
            raise TypeError("path must be PathEvidence")
        if self.kind == "path" and self.shape != "unknown":
            raise ValueError("A path cannot be assigned a primitive region shape")
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("Object ids must be nonnegative integers")
        if not isinstance(self.filled, bool):
            raise TypeError("filled must be boolean")
        if not (0 <= self.center_x <= 1 and 0 <= self.center_y <= 1):
            raise ValueError("Object centers must be normalized to [0, 1]")
        if not (0 < self.area <= 1 and 0 <= self.confidence <= 1):
            raise ValueError("Invalid area or confidence")
        for distribution, labels in ((self.shape_probabilities, SHAPES),
                                     (self.fill_probabilities, ("filled", "outline"))):
            if distribution is not None:
                if not distribution or any(k not in labels or not 0 <= v <= 1 for k, v in distribution.items()):
                    raise ValueError("Invalid attribute probability distribution")
                if abs(sum(distribution.values())-1) > 1e-4:
                    raise ValueError("Attribute probabilities must sum to one")

    def vertices(self) -> tuple[tuple[float, float], ...]:
        if self.kind == "path" and self.path is not None:
            return self.path.points
        if self.contour:
            return self.contour
        if self.shape == "unknown" or self.kind != "region":
            return ()
        n = {"circle": 32, "triangle": 3, "quadrilateral": 4}.get(self.shape, 5)
        # Area of a regular n-gon with circumradius r.
        radius = sqrt(2 * self.area / (n * sin(2 * pi / n)))
        start = (-pi / 2 if n != 4 else pi / 4) + self.angle * pi / 180
        return tuple((self.center_x + radius * cos(start + i * 2 * pi / n),
                      self.center_y + radius * sin(start + i * 2 * pi / n)) for i in range(n))


@dataclass
class Scene:
    objects: list[VisualObject] = field(default_factory=list)
    relations: set[tuple] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        ids = [obj.id for obj in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("Object ids must be unique within a scene")
        for relation in self.relations:
            if any(i not in ids for i in relation[1:]):
                raise ValueError("Relation references an unknown object")

    def to_dict(self):
        return {"objects": [asdict(o) for o in self.objects],
                "relations": sorted(self.relations), "warnings": self.warnings}

    @classmethod
    def from_dict(cls, data):
        objects = []
        for item in data["objects"]:
            item = dict(item)
            item["contour"] = tuple(tuple(p) for p in item.get("contour", ()))
            if item.get("path") is not None:
                item["path"] = PathEvidence.from_dict(item["path"])
            objects.append(VisualObject(**item))
        return cls(objects, {tuple(r) for r in data.get("relations", [])},
                   list(data.get("warnings", [])))

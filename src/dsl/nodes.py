"""Immutable typed AST; no eval(), executable source, or rule sentence lookup."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from src.scene.objects import Scene, VisualObject, SHAPES, COLORS
from src.scene.strokes import EVIDENCE_THRESHOLD, TOPOLOGIES

RELATIONS = ("left_of", "right_of", "above", "below", "inside", "touching",
             "overlapping", "near", "larger_than", "same_shape", "same_color", "same_size")


def _all(values):
    values = tuple(values)
    return False if False in values else None if None in values else True


def _any(values):
    values = tuple(values)
    return True if True in values else None if None in values else False


def _not(value):
    return None if value is None else not value


@dataclass(frozen=True)
class Predicate:
    op: str = "any"
    value: str = ""
    children: tuple[Predicate, ...] = ()

    def __post_init__(self):
        if self.op not in ("any", "shape", "color", "filled", "outline", "small", "large", "and", "or", "not",
                           "kind", "topology", "geometry", "has_arrows", "arrow_relation", "arrow_count",
                           "endpoint_count", "junction_count", "cycle_count"):
            raise ValueError(f"Unknown object predicate: {self.op}")
        if self.op == "shape" and self.value not in SHAPES:
            raise ValueError("Unknown shape")
        if self.op == "color" and self.value not in COLORS:
            raise ValueError("Unknown color")
        allowed = {"kind": ("region", "path", "unknown"), "topology": TOPOLOGIES,
                   "geometry": ("straight", "curved"), "arrow_relation": ("same", "opposed"),
                   "arrow_count": tuple(str(n) for n in range(9))}
        allowed.update({op:tuple(str(n) for n in range(9)) for op in ("endpoint_count","junction_count","cycle_count")})
        if self.op in allowed and self.value not in allowed[self.op]:
            raise ValueError("Unknown stroke predicate value")
        count = {"and": 2, "or": 2, "not": 1}.get(self.op, 0)
        if len(self.children) != count or any(not isinstance(c, Predicate) for c in self.children):
            raise TypeError("Invalid object predicate children")

    def __call__(self, obj: VisualObject):
        return self.assess(obj) is True

    def assess(self, obj: VisualObject) -> bool | None:
        """Hard predicates with explicit unavailable evidence, not invented labels."""
        if self.op == "and": return _all(c.assess(obj) for c in self.children)
        if self.op == "or": return _any(c.assess(obj) for c in self.children)
        if self.op == "not": return _not(self.children[0].assess(obj))
        if self.op == "any": return True
        if self.op == "kind":
            return None if obj.kind == "unknown" and self.value != "unknown" else obj.kind == self.value
        if self.op == "shape":
            return None if obj.kind != "region" or obj.shape == "unknown" else obj.shape == self.value
        if self.op in ("filled", "outline") and obj.kind != "region": return None
        if self.op in ("topology", "geometry", "has_arrows", "arrow_relation", "arrow_count",
                       "endpoint_count", "junction_count", "cycle_count"):
            if obj.kind == "region" and obj.path is None: return False
            if obj.kind == "unknown" or obj.path is None: return None
            path = obj.path
            if self.op == "topology":
                return None if path.topology == "unknown" or path.structural_confidence<EVIDENCE_THRESHOLD else path.topology == self.value
            if self.op in ("endpoint_count","junction_count","cycle_count"):
                count = path.graph_count(self.op.removesuffix("_count"))
                return None if count is None or path.structural_confidence<EVIDENCE_THRESHOLD else count==int(self.value)
            if self.op == "has_arrows":
                # A local arrow glyph can be known even when the whole graph
                # has no unique traversal. Absence needs a resolved path.
                if any(a.detected for a in path.arrows): return True
                return None if path.arrows or path.confidence<EVIDENCE_THRESHOLD else False
            if path.confidence < EVIDENCE_THRESHOLD: return None
            if self.op == "geometry": return None if path.geometry == "unknown" else path.geometry == self.value
            if self.op == "arrow_relation":
                return None if path.arrow_relation == "unknown" else path.arrow_relation == self.value
            if any(not a.detected for a in path.arrows): return None
            return len(path.arrows) == int(self.value)
        return {"color": obj.color == self.value,
                "filled": obj.filled, "outline": not obj.filled,
                "small": obj.area < 0.025, "large": obj.area >= 0.065}[self.op]

    @property
    def complexity(self):
        return 1 + sum(c.complexity for c in self.children)

    @property
    def text(self):
        if self.op in ("and", "or"):
            return f"({self.children[0].text} {self.op} {self.children[1].text})"
        if self.op == "not": return f"not {self.children[0].text}"
        return {"any": "object", "shape": self.value, "color": f"{self.value} object",
                "filled": "filled object", "outline": "outline object", "small": "small object",
                "large": "large object", "kind": self.value, "topology": f"{self.value} path",
                "geometry": f"{self.value} path", "has_arrows": "path with arrows",
                "arrow_relation": "path with arrows pointing "+("the same way along it" if self.value == "same" else "opposite ways along it"),
                "arrow_count": f"path with {self.value} arrows",
                "endpoint_count": f"path with {self.value} endpoints", "junction_count": f"path with {self.value} junctions",
                "cycle_count": f"path with {self.value} cycles"}[self.op]


@dataclass(frozen=True)
class Rule:
    op: str
    predicates: tuple[Predicate, ...] = ()
    children: tuple[Rule, ...] = ()
    n: int = 0
    relation: str = ""

    def __post_init__(self):
        arities = {"exists": (1, 0), "forall": (1, 0), "exactly": (1, 0), "atleast": (1, 0),
                   "pair": (2, 0), "forall_pair": (2, 0), "compare": (2, 0), "aligned": (1, 0),
                   "and": (0, 2), "or": (0, 2), "implies": (0, 2), "not": (0, 1)}
        if self.op not in arities:
            raise ValueError(f"Unknown rule operation: {self.op}")
        if (len(self.predicates), len(self.children)) != arities[self.op]:
            raise TypeError("Incorrect rule arity")
        if any(not isinstance(p, Predicate) for p in self.predicates) or any(not isinstance(c, Rule) for c in self.children):
            raise TypeError("Rule argument type mismatch")
        if self.op in ("exactly", "atleast") and (not isinstance(self.n, int) or self.n < 0):
            raise ValueError("Count must be a nonnegative integer")
        if self.op in ("pair", "forall_pair") and self.relation not in RELATIONS:
            raise ValueError("Unknown relation")
        if self.op == "compare" and self.relation not in ("gt", "lt", "eq", "approx"):
            raise ValueError("Unknown numerical comparison")

    @property
    def complexity(self):
        return 1 + sum(p.complexity for p in self.predicates) + sum(c.complexity for c in self.children)

    @property
    def depth(self):
        return 1 + max((c.depth for c in self.children), default=0)

    @property
    def key(self):
        return repr(self)

    def evaluate(self, scene: Scene) -> bool:
        return self.assess(scene) is True

    def assess(self, scene: Scene) -> bool | None:
        """Three-valued executable semantics; unknown cannot turn into a negative."""
        if self.op == "and": return _all(c.assess(scene) for c in self.children)
        if self.op == "or": return _any(c.assess(scene) for c in self.children)
        if self.op == "not": return _not(self.children[0].assess(scene))
        if self.op == "implies": return _any((_not(self.children[0].assess(scene)), self.children[1].assess(scene)))
        left = {o.id: self.predicates[0].assess(o) for o in scene.objects}
        lower = sum(v is True for v in left.values())
        upper = sum(v is not False for v in left.values())
        if self.op == "exists": return _any(left.values())
        if self.op == "forall": return _all(left.values())
        if self.op == "exactly":
            return False if self.n < lower or self.n > upper else True if lower == upper else None
        if self.op == "atleast":
            return True if lower >= self.n else False if upper < self.n else None
        if self.op == "compare":
            if self.predicates[0] == self.predicates[1]: return self.relation in ("eq", "approx")
            right = [self.predicates[1].assess(o) for o in scene.objects]
            rlower, rupper = sum(v is True for v in right), sum(v is not False for v in right)
            minimum, maximum = lower-rupper, upper-rlower
            if self.relation == "gt": return True if minimum > 0 else False if maximum <= 0 else None
            if self.relation == "lt": return True if maximum < 0 else False if minimum >= 0 else None
            margin = 1 if self.relation == "approx" else 0
            return True if -margin <= minimum <= maximum <= margin else False if minimum > margin or maximum < -margin else None
        if self.op == "aligned":
            return _any(_all((*[left[i] for i in ids], ("aligned", *sorted(ids)) in scene.relations))
                        for ids in combinations(left, 3))
        right = {o.id: self.predicates[1].assess(o) for o in scene.objects}
        entities = {obj.id: obj for obj in scene.objects}
        def relation_value(a, b):
            if a == b: return False
            if self.relation in ("inside", "touching", "overlapping") and (
                    entities[a].kind != "region" or entities[b].kind != "region" or
                    len(entities[a].vertices())<3 or len(entities[b].vertices())<3):
                return None
            if self.relation == "same_shape" and (entities[a].kind!="region" or entities[b].kind!="region" or
                                                   entities[a].shape=="unknown" or entities[b].shape=="unknown"):
                return None
            return (self.relation, a, b) in scene.relations
        matches = {a: _any(_all((value, relation_value(a, b)))
                           for b, value in right.items()) for a in left}
        if self.op == "pair": return _any(_all((left[a], matches[a])) for a in left)
        return _all(_any((_not(left[a]), matches[a])) for a in left)

    @classmethod
    def from_dict(cls, data, max_depth=12):
        if max_depth <= 0: raise ValueError("Program nesting limit exceeded")
        def predicate(item, remaining):
            if remaining <= 0: raise ValueError("Predicate nesting limit exceeded")
            return Predicate(item["op"], item.get("value", ""),
                             tuple(predicate(c, remaining-1) for c in item.get("children", [])))
        return cls(data["op"], tuple(predicate(p, max_depth) for p in data.get("predicates", [])),
                   tuple(cls.from_dict(c, max_depth-1) for c in data.get("children", [])),
                   data.get("n", 0), data.get("relation", ""))

    def explain(self, scene):
        from .verbalizer import verbalize
        assessment = self.assess(scene)
        evidence = {"matches": assessment, "assessment": "unknown" if assessment is None else "true" if assessment else "false",
                    "rule": verbalize(self),
                    "objects": [{"id": o.id, "kind": o.kind, "shape": o.shape, "filled": o.filled,
                                 **({"topology": o.path.topology, "arrow_relation": o.path.arrow_relation,
                                     "direction_confidence": o.path.direction_confidence,
                                     "arrows": [{"arc_position": a.arc_position, "direction": a.direction,
                                                 "confidence": a.confidence,
                                                 "presence_confidence": a.presence_confidence} for a in o.path.arrows]} if o.path else {})}
                                for o in scene.objects]}
        if self.children:
            evidence["steps"] = [c.explain(scene) for c in self.children]
        else:
            selected = [o.id for o in scene.objects if self.predicates[0](o)]
            evidence["selected_ids"] = selected
            evidence["unknown_ids"] = [o.id for o in scene.objects if self.predicates[0].assess(o) is None]
            evidence["count"] = len(selected)
            if self.op in ("pair", "forall_pair"):
                right = [o.id for o in scene.objects if self.predicates[1](o)]
                evidence["relation"] = self.relation
                evidence["witnesses"] = [(a, b) for a in selected for b in right if a != b and (self.relation, a, b) in scene.relations]
                entities = {o.id:o for o in scene.objects}
                def unresolved(a,b):
                    if self.relation=="same_shape":
                        return entities[a].kind!="region" or entities[b].kind!="region" or entities[a].shape=="unknown" or entities[b].shape=="unknown"
                    if self.relation in ("inside","touching","overlapping"):
                        return entities[a].kind!="region" or entities[b].kind!="region" or len(entities[a].vertices())<3 or len(entities[b].vertices())<3
                    return False
                uncertain = [a for a in selected if any(unresolved(a,b) for b in right if a!=b)]
                evidence["uncertain_relation_ids"] = uncertain
                evidence["violating_ids"] = [a for a in selected if a not in uncertain and not any((a, b) in evidence["witnesses"] for b in right)]
        return evidence


def Count(n, predicate=Predicate(), at_least=False):
    return Rule("atleast" if at_least else "exactly", (predicate,), n=n)


def Exists(predicate): return Rule("exists", (predicate,))
def ForAll(predicate): return Rule("forall", (predicate,))
def Relation(name, left=Predicate(), right=Predicate(), universal=False):
    return Rule("forall_pair" if universal else "pair", (left, right), relation=name)
def Compare(left, right, operation="gt"): return Rule("compare", (left, right), relation=operation)


def _binary(op, a, b):
    if op in ("and", "or"):
        if a == b: return a
        a, b = sorted((a, b), key=lambda r: r.key)
    return Rule(op, children=(a, b))


def And(a, b): return _binary("and", a, b)
def Or(a, b): return _binary("or", a, b)
def Not(rule): return rule.children[0] if rule.op == "not" else Rule("not", children=(rule,))


def uses_paths(rule):
    """Whether a program needs graph evidence, including composed selectors."""
    path_ops = {"kind", "topology", "geometry", "has_arrows", "arrow_relation", "arrow_count",
                "endpoint_count", "junction_count", "cycle_count"}
    def predicate_uses_paths(predicate):
        return predicate.op in path_ops or any(predicate_uses_paths(child) for child in predicate.children)
    return any(predicate_uses_paths(predicate) for predicate in rule.predicates) or any(uses_paths(child) for child in rule.children)

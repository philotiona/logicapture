"""Constructive graph-scene examples for the shared symbolic rule language.

The domain varies centerline geometry independently of arrow signs.  Direction
counterexamples retain exactly the same path and change one arrowhead.  Every
returned drawing is checked by the executable rule, including unknown evidence.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations, product
from math import ceil, cos, pi, sin, acos
from random import Random

from src.scene import ArrowMark, PathEvidence, Scene, VisualObject, derive_relations
from src.dsl.nodes import uses_paths
from .scene_generation import GenerationError


def _centerline(family):
    if family == "line": return tuple((.15+.7*t/80, .5) for t in range(81))
    if family == "arc": return tuple((.5+.33*cos(pi/4+t*1.5*pi/100), .5+.33*sin(pi/4+t*1.5*pi/100)) for t in range(101))
    if family == "s": return tuple((.15+.7*t/100, .5+.22*sin(2*pi*t/100)) for t in range(101))
    if family == "wave": return tuple((.15+.7*t/100, .5+.19*sin(3*pi*t/100)) for t in range(101))
    if family == "loop": return tuple((.5+.32*cos(2*pi*t/100), .5+.24*sin(2*pi*t/100)) for t in range(101))
    if family == "irregular_loop": return tuple((.5+(.25+.045*sin(3*2*pi*t/100))*cos(2*pi*t/100),
                                                  .5+(.28+.035*cos(5*2*pi*t/100))*sin(2*pi*t/100)) for t in range(101))
    if family == "branch": return ((.5,.15),(.5,.5),(.2,.75),(.5,.5),(.8,.75))
    raise ValueError("Unknown centerline family")


def path_object(family="s", directions=(1, -1), object_id=0):
    """A known symbolic path; directions are independent of its geometry."""
    topology = "closed" if "loop" in family else "branched" if family == "branch" else "open"
    points = _centerline(family)
    geometry = PathEvidence(points,topology)
    def attachment_candidates(nominal):
        # Place marks on a locally legible section, independent of arrow sign or
        # group label. A triangle covering a tight U-bend is an ambiguous glyph.
        window = min(.14,.38/(len(directions)+1))
        positions = [max(.10,min(.90,nominal-window+2*window*i/30)) for i in range(31)]
        def turn(position):
            before = geometry.point_tangent(max(0.,position-.035))[1]
            after = geometry.point_tangent(min(1.,position+.035))[1]
            return acos(max(-1.,min(1.,before[0]*after[0]+before[1]*after[1])))+.08*abs(position-nominal)
        return [(turn(position),position) for position in positions]
    layouts=[(0.,())]
    for index in range(len(directions)):
        candidates=attachment_candidates((index+1)/(len(directions)+1))
        updated=[]
        for score,position in candidates:
            compatible=[(previous,layout) for previous,layout in layouts if not layout or position-layout[-1]>=.14]
            if compatible:
                previous,layout=min(compatible,key=lambda item:item[0])
                updated.append((previous+score,(*layout,position)))
        layouts=updated
    positions=min(layouts,key=lambda item:item[0])[1] if layouts else ()
    marks = tuple(ArrowMark(position, sign, 1.) for position,sign in zip(positions,directions))
    path = PathEvidence(points, topology, marks, 1., width=.008,
                        endpoint_count=3 if topology=="branched" else 0 if topology=="closed" else 2,
                        junction_count=1 if topology=="branched" else 0,
                        cycle_count=1 if topology=="closed" else 0,
                        endpoints=(points[0],points[2],points[-1]) if topology=="branched" else () if topology=="closed" else (points[0],points[-1]),
                        junctions=(points[1],) if topology=="branched" else (),
                        segments=((points[0],points[1]),(points[1],points[2]),(points[1],points[-1])) if topology=="branched" else ())
    return VisualObject(object_id, "unknown", False, area=max(.001, path.length*path.width),
                        kind="path", path=path, attribute_source="symbolic")


def flip_arrow(scene, object_id=0, arrow_index=0):
    objects = []
    for obj in scene.objects:
        if obj.id != object_id or obj.path is None: objects.append(obj); continue
        marks = list(obj.path.arrows)
        marks[arrow_index] = replace(marks[arrow_index], direction=-marks[arrow_index].direction,
                                    probabilities=None)
        objects.append(replace(obj, path=replace(obj.path, arrows=tuple(marks))))
    return derive_relations(Scene(objects))


def transform_path(path, angle=0., mirror=False, scale=1., offset=(0., 0.)):
    """Rigid pose changes preserve signs relative to path traversal."""
    theta = angle*pi/180
    def point(p):
        x, y = p[0]-.5, p[1]-.5
        if mirror: x = -x
        return (.5+scale*(x*cos(theta)-y*sin(theta))+offset[0],
                .5+scale*(x*sin(theta)+y*cos(theta))+offset[1])
    marks = tuple(replace(a, tip=point(a.tip) if a.tip else None, tail=point(a.tail) if a.tail else None)
                  for a in path.arrows)
    return replace(path, points=tuple(point(p) for p in path.points), arrows=marks, width=path.width*scale,
                   endpoints=tuple(point(p) for p in path.endpoints),junctions=tuple(point(p) for p in path.junctions),
                   segments=tuple(tuple(point(p) for p in segment) for segment in path.segments))


class PathSceneGenerator:
    def __init__(self, seed=42, color=False, topologies=("open","closed","branched")):
        if not topologies or any(value not in ("open","closed","branched") for value in topologies):
            raise ValueError("Unsupported generator topology domain")
        self.rng = Random(seed)
        self.color = color
        self._templates = [path_object(family, signs)
                           for family, signs in product(("line", "arc", "s", "wave", "loop", "irregular_loop", "branch"),
                                                        ((), (1,), (1, 1), (1, -1), (1, 1, 1), (1, -1, 1)))]
        self._templates = [obj for obj in self._templates if obj.path.topology in topologies]
        self._templates += [VisualObject(0, shape, filled) for shape, filled in
                            product(("circle", "triangle", "quadrilateral"), (True, False))]

    def _pack(self, objects):
        count = len(objects)
        if count == 0: return Scene()
        columns = 1 if count == 1 else 2 if count <= 4 else 3
        rows = ceil(count/columns)
        slots = [((x+.5)/columns, (y+.5)/rows) for y in range(rows) for x in range(columns)]
        self.rng.shuffle(slots)
        packed = []
        scale = .9 if count == 1 else .8/max(columns, rows)
        for i, (obj, (cx, cy)) in enumerate(zip(objects, slots)):
            path = transform_path(obj.path, self.rng.uniform(-180,180), self.rng.choice((True,False)),
                                  scale, (cx-.5, cy-.5)) if obj.path else None
            packed.append(replace(obj, id=i, center_x=cx, center_y=cy, path=path,
                                  area=max(.001,obj.area*scale*scale)))
        return derive_relations(Scene(packed))

    def _selector_scene(self, predicate, matches, distractors=0):
        if not 0 <= matches+distractors <= 8: raise GenerationError("Path scenes support at most eight entities")
        yes = [obj for obj in self._templates if predicate.assess(obj) is True]
        no = [obj for obj in self._templates if predicate.assess(obj) is False]
        def needs_traversal(p):
            return p.op in ("has_arrows","arrow_relation","arrow_count") or any(needs_traversal(c) for c in p.children)
        if needs_traversal(predicate):
            # Arrow interpretation currently requires a unique traversal.
            # Branch topology is generated for graph features, not presented as
            # a solved direction/presence example outside that perception scope.
            yes = [obj for obj in yes if obj.path is None or obj.path.topology!="branched"]
            no = [obj for obj in no if obj.path is None or obj.path.topology!="branched"]
        # For graph features, keep the entity domain the same in both groups.
        # An open/closed comparison should not teach a path/triangle shortcut.
        path_yes, path_no = [obj for obj in yes if obj.kind=="path"],[obj for obj in no if obj.kind=="path"]
        if path_yes and path_no:
            yes,no = path_yes,path_no
        if matches and not yes or distractors and not no:
            raise GenerationError("Selector is outside the constructive path domain")
        return self._pack([self.rng.choice(yes) for _ in range(matches)] +
                          [self.rng.choice(no) for _ in range(distractors)])

    def sample(self, rule, label, max_attempts=4000, count=None):
        label = bool(label)
        # Construct counts over selectors, rather than hoping a random drawing
        # happens to meet a rare concept.  Missing evidence cannot be negative.
        if rule.op in ("exactly", "atleast", "exists", "forall"):
            if rule.op == "forall": matches, distractors = ((1,0) if label else (0,1))
            elif rule.op == "exists": matches, distractors = ((1,0) if label else (0,1))
            else:
                matches = rule.n if label else max(0,rule.n-1) if rule.n else 1
                distractors = 0
            if count is not None: distractors = count-matches
            try:
                scene = self._selector_scene(rule.predicates[0], matches, distractors)
                if rule.assess(scene) is label: return scene
            except GenerationError: pass
        # A bounded constructive domain supports arbitrary AND/OR/NOT graphs.
        # It is enumerated before choosing a pose; no unvalidated fallback.
        candidates = [Scene()]
        candidates += [Scene([obj]) for obj in self._templates]
        candidates += [Scene([replace(obj,id=i) for i in range(n)]) for obj in self._templates for n in (2,3,4,5,6)]
        candidates += [Scene([a,replace(b,id=1)]) for a,b in combinations(self._templates,2)]
        supported = [scene for scene in candidates if (count is None or len(scene.objects)==count) and rule.assess(scene) is label]
        if not supported: raise GenerationError("No checked example exists in the constructive path domain")
        self.rng.shuffle(supported)
        for scene in supported:
            packed = self._pack(scene.objects)
            if rule.assess(packed) is label: return packed
        raise GenerationError("The rule depends on geometry outside the constructive path domain")

    def balanced_examples(self, rule, per_group=6):
        positives, negatives = [], []
        for _ in range(per_group):
            positive = self.sample(rule, True)
            negative = None
            # Counterfactual edits isolate arrow direction from all shape cues.
            for obj in positive.objects:
                if obj.path is None: continue
                for index in range(len(obj.path.arrows)):
                    candidate = flip_arrow(positive,obj.id,index)
                    if rule.assess(candidate) is False: negative = candidate; break
                if negative is not None: break
            if negative is None: negative = self.sample(rule,False)
            positives.append((positive,True)); negatives.append((negative,False))
        return positives+negatives

    def pool(self, size=280):
        """A reproducible disagreement domain spanning factored graph features."""
        domain = [Scene()]
        domain += [self._pack([obj]) for obj in self._templates]
        domain += [self._pack([obj]*n) for obj in self._templates for n in (2,3,4)]
        domain += [self._pack([a,b]) for a,b in combinations(self._templates,2)]
        if size is not None and size < len(domain):
            # Retain every singleton feature before sampling mixed examples.
            essentials = 1+len(self._templates)
            tail = domain[essentials:]
            self.rng.shuffle(tail)
            domain = domain[:essentials]+tail[:max(0,size-essentials)]
        return domain

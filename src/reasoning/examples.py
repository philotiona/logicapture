"""Construct new drawings and re-read their pixels against every frozen rule."""
from dataclasses import dataclass
from math import pi
import cv2
import numpy as np

from src.dsl.nodes import Rule, uses_paths
from src.active import SceneGenerator, GenerationError
from src.active.path_generation import PathSceneGenerator
from src.scene.renderer import draw_scene
from src.vision import extract_scene
from src.vision.observations import observations
from .measurement_rules import MeasurementRule


@dataclass
class CheckedExample:
    image: np.ndarray
    label: bool
    verification: dict


def _dotted(rng, spacing, size=320):
    image = np.full((size,size,3),255,np.uint8)
    radius = int(rng.integers(2,5))
    # Arc-length resampling maintains proximity through bends and rotations.
    t = np.linspace(0,1,1000)
    family = int(rng.integers(0,3))
    x = .15+.7*t
    y = .5+(.23*np.sin(2*pi*t) if family == 0 else .28*np.sin(pi*t) if family == 1 else .2*(t-.5))
    points = np.column_stack((x,y))*size
    lengths = np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    target = np.arange(0,lengths[-1],max(radius*2.3,spacing*(2*radius+1)))
    sampled = np.column_stack([np.interp(target,lengths,points[:,axis]) for axis in (0,1)])
    angle = float(rng.uniform(0,2*pi))
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    sampled = (sampled-size/2)@rotation+size/2
    for x,y in sampled:
        cv2.rectangle(image,(round(x-radius),round(y-radius)),(round(x+radius),round(y+radius)),(0,0,0),-1)
    return image


def _boundary_drawing(rng, size=320, *, concave=None, elongated=None, filled=None, corners=None):
    """New convex/concave and elongated figures, checked after rendering."""
    image = np.full((size, size, 3), 255, np.uint8)
    corners = int(rng.integers(3, 9)) if corners is None else corners
    concave = bool(rng.integers(2)) if concave is None else concave
    angles = np.linspace(0, 2*pi, corners*(2 if concave else 1), endpoint=False)
    radius = np.ones(len(angles))
    if concave:
        # Stay inside the apothem even for a three-point star; otherwise the
        # supposed concave example could still be a convex hexagon.
        radius[1::2] = float(rng.uniform(.18, .30))
    points = np.column_stack((np.cos(angles), np.sin(angles)))*radius[:, None]
    aspect = rng.uniform(.12, .23) if elongated is True else rng.uniform(.8, 1.) if elongated is False else rng.uniform(.12, 1.)
    points[:, 0] *= float(aspect)
    points *= float(rng.uniform(.24, .36))*size
    angle = float(rng.uniform(0, 2*pi))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.rint(points@rotation+size/2+rng.uniform(-.06, .06, 2)*size).astype(np.int32)
    filled = bool(rng.integers(2)) if filled is None else filled
    if filled:
        cv2.fillPoly(image, [points], (0, 0, 0))
    else:
        cv2.polylines(image, [points], True, (0, 0, 0), int(rng.integers(2, 5)))
    return image


def geometric_demo(concept, *, seed=73, per_group=6):
    """A controlled pixel-only demonstration, not an external benchmark.

    The concept names select what the generator draws. Only the resulting
    pixels and group membership enter the solver, exactly as with an upload.
    """
    if concept not in ("convexity", "elongation") or not 4 <= per_group <= 8:
        raise ValueError("Choose convexity or elongation and 4–8 examples per group.")
    rng = np.random.default_rng(seed)
    groups = []
    for label in (True, False):
        groups.append([_boundary_drawing(rng,
            concave=not label if concept == "convexity" else bool(index % 2),
            elongated=label if concept == "elongation" else False,
            filled=bool((index if concept == "convexity" else index//2) % 2),
            corners=3+index % 5) for index in range(per_group)])
    return tuple(groups)


def generate_checked_examples(diagnostics, *, seed=97, attempts=64):
    candidates = diagnostics.get("structural_attempt",{}).get("candidates",[])
    if not candidates:
        raise GenerationError("No executable rule is available for checked generation")
    programs = [candidate["program"] for candidate in candidates]
    numeric = programs[0]["op"] == "measurement_threshold"
    rules = ([MeasurementRule(p["feature"],p["threshold"],p["positive_below"]) for p in programs] if numeric
             else [Rule.from_dict(p) for p in programs])
    rng = np.random.default_rng(seed)
    generator = PathSceneGenerator(seed=seed) if not numeric and any(uses_paths(r) for r in rules) else SceneGenerator(seed=seed)
    examples = []
    for label in (True,False):
        for iteration in range(attempts):
            try:
                if numeric and rules[0].feature.startswith("dot_spacing"):
                    image = _dotted(rng, float(rng.uniform(1.15,4.5)))
                elif numeric and any("boundary_convexity" in rule.feature or "elongation" in rule.feature for rule in rules):
                    image = _boundary_drawing(rng)
                elif numeric:
                    # Broad constructive pool, never a copied support drawing.
                    image = np.array(draw_scene(generator.random_scene(int(rng.integers(0,7)))))
                else:
                    image = np.array(draw_scene(generator.sample(rules[0],label,2000)))
                evidence = observations(image) if numeric else extract_scene(image)
                votes = [rule.assess(evidence) for rule in rules]
                if all(v is label for v in votes):
                    examples.append(CheckedExample(image,label,{"pixel_reread":True,"rule_votes":votes,
                        "program_count":len(rules),"generator_seed":seed,"attempt":iteration+1}))
                    break
            except GenerationError:
                continue
        else:
            raise GenerationError("Could not construct and pixel-verify both classes within the generation budget")
    return examples

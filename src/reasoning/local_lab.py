"""CPU-only contrast learning and compositional drawing generation."""
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.vision.adaptive_perception import ink_mask, remove_ruling
from src.vision.pretrained import image_digest, rgb_image
from src.vision.structural_features import StructuralEncoder
from .episode import AbstentionPolicy, Candidate, EpisodeClassifier, EpisodePrediction, FittedEpisode
from .features import FeatureBatch


SIDE = 128
FAMILIES = ("component count", "hole count", "topology", "boundary", "arrangement", "symmetry")
NEAR_COPY = .90


def _panel(image, framed):
    rgb = np.array(ImageOps.pad(rgb_image(image), (SIDE, SIDE),
                               method=Image.Resampling.LANCZOS, color="white"))
    ink = ink_mask(rgb)
    if framed:
        ink, _ = remove_ruling(ink)
    if not .001 <= np.mean(ink > 0) <= .45:
        raise ValueError("Use a clear drawing with dark marks on a light background.")
    # Remove paper tint and compression noise; keep the positions and colors of ink.
    rgb[ink == 0] = 255
    return rgb


def _ink(pixels):
    return np.any(pixels < 245, axis=2)


def _silhouette(pixels):
    ink = _ink(pixels)
    y, x = np.nonzero(ink)
    mask = Image.fromarray(np.uint8(ink[y.min():y.max()+1, x.min():x.max()+1])*255)
    return np.asarray(ImageOps.pad(mask, (72, 72), method=Image.Resampling.NEAREST, color=0)) > 0


def _similarity(left, right):
    return np.count_nonzero(left & right) / np.count_nonzero(left | right)


class _LabEncoder:
    def __init__(self):
        self.structural = StructuralEncoder(side=96, strict=True)

    def encode(self, panels, ids):
        encoded = self.structural.encode(panels)
        schema = encoded.as_feature_batch(ids, append_validity=False).metadata["feature_families"]
        schema["component count"] = [encoded.names.index("invariant.component_count")]
        schema["hole count"] = [encoded.names.index("invariant.hole_count")]
        views = {name: encoded.values[:, schema[name]] for name in FAMILIES}
        return FeatureBatch(tuple(ids), np.column_stack(list(views.values())),
                            encoder_id="local-lab-structural-v1", extra_views=views)


def _candidates(positive, references):
    shapes = [_silhouette(panel) for panel in references]
    seen = {image_digest(panel) for panel in references}
    for i, j in combinations(range(len(positive)), 2):
        for axis in (0, 1):
            left, right = positive[i], positive[j]
            if axis:
                left, right = left.transpose(1, 0, 2), right.transpose(1, 0, 2)
            li, ri = _ink(left), _ink(right)
            # Both parents must have a clear corridor around the cut.
            empty = ~(li.any(axis=1) | ri.any(axis=1))
            cuts = [row for row in range(SIDE//4, 3*SIDE//4)
                    if np.all(empty[row-2:row+2])]
            if not cuts:
                continue
            cut = min(cuts, key=lambda row: abs(row-SIDE//2))
            for upper, lower, ui, di, a, b in ((left, right, li, ri, i, j), (right, left, ri, li, j, i)):
                if min(np.count_nonzero(ui[:cut]), np.count_nonzero(di[cut:])) < 12:
                    continue
                rendered = np.concatenate((upper[:cut], lower[cut:]), axis=0)
                if axis:
                    rendered = rendered.transpose(1, 0, 2)
                digest = image_digest(rendered)
                if digest in seen:
                    continue
                seen.add(digest)
                silhouette = _silhouette(rendered)
                similarity = max(_similarity(silhouette, reference) for reference in shapes)
                if similarity < NEAR_COPY:
                    yield rendered, (f"A{a+1:02}", f"A{b+1:02}"), similarity


@dataclass
class LabResult:
    fit: FittedEpisode
    encoder: _LabEncoder
    reference_hashes: frozenset[str]
    candidate: np.ndarray | None = None
    donors: tuple[str, str] | None = None
    similarity: float | None = None
    decision: EpisodePrediction | None = None
    tried: int = 0
    reason: str = ""

    def check(self, image, *, framed=False):
        panel = _panel(image, framed)
        if image_digest(panel) in self.reference_hashes:
            raise ValueError("This repeats a reference. Use a drawing the lab has not seen.")
        return self.fit.predict(self.encoder.encode([panel], ["Test"]))[0]


def learn_and_draw(positive, negative, *, framed=False, progress=None):
    if not all(3 <= len(group) <= 8 for group in (positive, negative)):
        raise ValueError("Use 3–8 different drawings per group.")
    pos, neg = ([_panel(image, framed) for image in group] for group in (positive, negative))
    refs = pos + neg
    hashes = frozenset(map(image_digest, refs))
    if len(hashes) != len(refs):
        raise ValueError("Use distinct drawings in both groups.")
    encoder = _LabEncoder()
    ids = [f"{label}{index+1:02}" for label, group in (("A", pos), ("B", neg)) for index in range(len(group))]
    if progress:
        progress("Learning a contrast and holding out reference drawings")
    choices = tuple(Candidate(view, "support_standardize" if view.endswith(" count") else "support_contrast",
                              learner, .1, 6)
                    for view in FAMILIES for learner in ("prototype", "rbf"))
    classifier = EpisodeClassifier(candidates=choices, policy=AbstentionPolicy(
        minimum_cv_accuracy=.75, minimum_agreement=.9))
    fit = classifier.fit(encoder.encode(refs, ids), [True]*len(pos)+[False]*len(neg))
    result = LabResult(fit, encoder, hashes)
    if fit.diagnostics["contradictory_pairs"] or fit.diagnostics["nested_cv_accuracy"] < fit.policy.minimum_cv_accuracy:
        result.reason = "The learned contrast did not hold up on the reference checks."
        return result
    if progress:
        progress("Recombining whole parts and checking the rendered drawings")
    candidates = list(_candidates(pos, refs))
    result.tried = len(candidates)
    if not candidates:
        result.reason = "No sufficiently new drawing could be made by recombining whole parts."
        return result
    batch = encoder.encode([item[0] for item in candidates], [f"New{n+1}" for n in range(len(candidates))])
    predictions = fit.predict(batch)
    accepted = [(decision.diagnostics["nearest_support_distance"], -decision.score, index)
                for index, decision in enumerate(predictions) if decision.label is True]
    if not accepted:
        result.reason = "None of the recombined drawings passed the Group A checks."
        return result
    index = min(accepted)[2]
    result.candidate, result.donors, result.similarity = candidates[index]
    result.decision = predictions[index]
    return result


def lab_demo(*, seed=12, per_group=6):
    """A controlled input with independent rendering seeds and no rule labels."""
    rng = np.random.default_rng(seed)
    groups = []
    for label in (True, False):
        images = []
        for index in range(per_group):
            image = np.full((SIDE, SIDE, 3), 255, np.uint8)
            # Distribute objects across both bands. Scale, placement and stroke
            # style vary in each group independently of the count.
            top = 1 + index % 2
            bottom = 3-top if label else (2 if index % 2 else 1)
            for row, count in ((32, top), (92, bottom)):
                slots = rng.choice((25, 63, 103), count, replace=False)
                for slot in slots:
                    x, y = int(slot+rng.integers(-4, 5)), int(row+rng.integers(-7, 8))
                    radius = int(rng.integers(6, 13))
                    cv2.circle(image, (x, y), radius, (15, 15, 15), int(rng.choice((-1, 2))))
            images.append(image)
        groups.append(images)
    return tuple(groups)

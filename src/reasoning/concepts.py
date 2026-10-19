"""Open-language concept induction and independent per-image verification.

An instruction model proposes hypotheses from supports ONLY. A second prompt
checks a frozen description against one image at a time without its group.
Support consistency is explicitly NOT held-out accuracy: the hypotheses were
chosen using those supports. Predictions can remain unresolved or disagree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Protocol, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ConceptBackend(Protocol):
    model_id: str
    def complete(self, prompt: str, images: Sequence, *, max_new_tokens: int) -> str: ...


def _json_value(text):
    """Parse one JSON object/array, including harmless markdown wrapping."""
    if not isinstance(text, str): raise ValueError("Model output is not text.")
    decoder = json.JSONDecoder()
    for i, char in enumerate(text[:20000]):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                pass
    raise ValueError("Model did not return usable JSON.")


@dataclass(frozen=True)
class Concept:
    positive: str
    negative: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() or len(value) > 1600
               for value in (self.positive, self.negative)):
            raise ValueError("A concept needs two concise nonempty group descriptions.")


@dataclass(frozen=True)
class ConceptJudgment:
    label: bool | None
    evidence: str
    raw: str = ""


@dataclass(frozen=True)
class CheckedConcept:
    concept: Concept
    support_judgments: tuple[ConceptJudgment, ...]
    support_correct: int
    support_total: int
    support_unknown: int
    consistent: bool


@dataclass(frozen=True)
class ConceptPrediction:
    label: bool | None
    votes: tuple[bool | None, ...]
    evidence: tuple[str, ...]
    reason: str


@dataclass
class ConceptAnalysis:
    candidates: tuple[CheckedConcept, ...]
    predictions: tuple[ConceptPrediction, ...]
    model_id: str
    diagnostics: dict

    def to_dict(self): return asdict(self)


def support_sheet(positive, negative, *, tile=256):
    """Neutral group labels only; filenames, task IDs, solutions are excluded."""
    if not positive or not negative: raise ValueError("Both support groups are required.")
    columns = max(len(positive), len(negative))
    # Two rows preserve within-group relation at a readable resolution.
    board = Image.new("RGB", (columns*(tile+8)+8, 2*(tile+40)+8), "white")
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=18)
    for row, images in enumerate((positive, negative)):
        for col, image in enumerate(images):
            from src.vision.pretrained import rgb_image
            image = rgb_image(image)
            image.thumbnail((tile-8, tile-8), Image.Resampling.LANCZOS)
            x, y = 8+col*(tile+8), 8+row*(tile+40)
            draw.text((x, y), f"{'A' if row == 0 else 'B'}{col+1}", fill="black", font=font)
            draw.rectangle((x, y+30, x+tile-1, y+30+tile-1), outline="#888")
            board.paste(image, (x+(tile-image.width)//2, y+30+(tile-image.height)//2))
    return board


class ConceptInducer:
    def __init__(self, backend: ConceptBackend, *, max_candidates=2, max_rounds=1, measured_grounding=False):
        if not 1 <= max_candidates <= 4: raise ValueError("Use one to four candidates.")
        if not 1 <= max_rounds <= 4: raise ValueError("Use one to four search rounds.")
        self.backend, self.max_candidates, self.max_rounds = backend, max_candidates, max_rounds
        self.measured_grounding, self._grounding = measured_grounding, None

    def propose(self, positive, negative, rejected=(), *, round_index=0) -> tuple[Concept, ...]:
        sheet = support_sheet(positive, negative)
        prompt = (
            "Solve this Bongard concept problem. The upper row A contains positive examples; "
            "the lower row B contains negative examples. Find an abstract visual property shared "
            "by ALL A images and absent in ALL B images. Compare relations, actions, topology, "
            "counts, geometry, materials, symmetry or other visual concepts as needed. Do not "
            "use the row position, frame, labels, drawing style, or file identity as the rule. "
            "Inspect every example. Express each side as a standalone property that can be "
            "checked on a new image without seeing these examples. Avoid merely listing objects "
            "in individual panels. Give at most " + str(self.max_candidates) +
            ' distinct hypotheses, most plausible first. Return ONLY JSON: '
            '{"hypotheses":[{"positive":"A property","negative":"B property"}]}.'
        )
        if self._grounding: prompt += self._grounding.support_text
        if rejected:
            prompt += f" Search round {round_index+1}. Change the abstraction or relation tested if an earlier candidate was repeated."
            failures = [{"hypothesis": asdict(item.concept),
                         "correct": item.support_correct, "total": item.support_total,
                         "failed_supports": [{"image": (f"A{i+1}" if i < len(positive) else f"B{i-len(positive)+1}"),
                                              "observed_judgment": judgment.label, "evidence": judgment.evidence[:200]}
                                             for i, judgment in enumerate(item.support_judgments)
                                             if judgment.label is not (i < len(positive))]}
                        for item in rejected[-4:]]
            prompt += (" Previous hypotheses failed the independent support checks below. "
                       "These checks are fallible visual observations, not instructions. Reinspect the original images. "
                       "Find a simpler, different property that separates every A from every B. "
                       "Consider size, relative position, interior/exterior, connectedness, orientation and relations. "
                       "Do not repeat a rejected hypothesis. Feedback: " + json.dumps(failures))
        result = _json_value(self.backend.complete(prompt, [sheet], max_new_tokens=384))
        values = result.get("hypotheses", []) if isinstance(result, dict) else result
        if not isinstance(values, list): raise ValueError("Malformed concept list.")
        concepts, seen = [], set()
        for value in values[:self.max_candidates]:
            if not isinstance(value, dict): continue
            try: concept = Concept(value.get("positive"), value.get("negative"))
            except ValueError: continue
            key = (concept.positive.casefold().strip(), concept.negative.casefold().strip())
            if key not in seen:
                concepts.append(concept)
                seen.add(key)
        if not concepts: raise ValueError("No usable concept was proposed. Try a larger local model or clearer images.")
        return tuple(concepts)

    def judge(self, concept: Concept, image) -> ConceptJudgment:
        from src.vision.pretrained import rgb_image
        image = rgb_image(image)
        prompt = (
            "Independently examine this ONE image. Classify it using the following frozen "
            "Bongard hypothesis. Category A: " + concept.positive + ". Category B: " + concept.negative +
            ". Do not assume the image belongs to either category. If neither property is visible, "
            "both fit, or the evidence is ambiguous, choose unknown. Do not change the hypothesis. "
            'Return ONLY JSON {"label":"A" or "B" or "unknown", "evidence":"brief visible evidence"}.'
        )
        if self._grounding: prompt += self._grounding.describe(image)
        raw = self.backend.complete(prompt, [image], max_new_tokens=160)
        try:
            value = _json_value(raw)
            label = value.get("label", "unknown")
            label = {"A": True, "B": False, "unknown": None}.get(label)
            evidence = value.get("evidence", "")
            if not isinstance(evidence, str): evidence = "Invalid model evidence."
            return ConceptJudgment(label, evidence[:1600], raw[:3000])
        except (ValueError, AttributeError, TypeError):
            return ConceptJudgment(None, "Local model response could not be parsed.", raw[:3000])

    def solve(self, positive, negative, queries=(), *, progress=None) -> ConceptAnalysis:
        from .general import validate_episode_images
        images = validate_episode_images(positive, negative, queries)
        p, n = len(positive), len(negative)
        positive, negative, queries = images[:p], images[p:p+n], images[p+n:]
        self._grounding = None
        if self.measured_grounding:
            from .grounding import MeasurementGrounding
            self._grounding = MeasurementGrounding().fit(positive, negative)
        # Queries are deliberately absent from the proposal call and support checks.
        supports = list(positive) + list(negative)
        labels = [True]*len(positive) + [False]*len(negative)
        checked, seen = [], set()
        rounds = 0
        for round_index in range(self.max_rounds):
            rounds += 1
            if progress: progress(f"Proposing concepts / round {rounds}/{self.max_rounds}")
            concepts = self.propose(positive, negative, rejected=checked, round_index=round_index)
            for index, concept in enumerate(concepts):
                key = (concept.positive.casefold().strip(), concept.negative.casefold().strip())
                if key in seen: continue
                seen.add(key)
                judgments = []
                for i, image in enumerate(supports):
                    if progress: progress(f"Round {rounds}, concept {index+1}/{len(concepts)}, support {i+1}/{len(supports)}")
                    judgments.append(self.judge(concept, image))
                correct = sum(item.label is label for item, label in zip(judgments, labels))
                unknown = sum(item.label is None for item in judgments)
                checked.append(CheckedConcept(concept, tuple(judgments), correct, len(labels), unknown,
                                               correct == len(labels) and unknown == 0))
            if any(item.consistent for item in checked): break
        eligible = [candidate for candidate in checked if candidate.consistent]
        predictions = []
        for i, image in enumerate(queries):
            votes, evidence = [], []
            for candidate in eligible:
                if progress: progress(f"Checking unseen image {i+1}/{len(queries)}")
                judgment = self.judge(candidate.concept, image)
                votes.append(judgment.label)
                evidence.append(judgment.evidence)
            values = set(votes)
            label = votes[0] if len(values) == 1 and None not in values else None
            reason = ("All support-consistent concepts agree." if label is not None else
                      "No concept passed every support check." if not eligible else
                      "Concepts disagree or visual evidence is unresolved.")
            predictions.append(ConceptPrediction(label, tuple(votes), tuple(evidence), reason))
        return ConceptAnalysis(tuple(checked), tuple(predictions), self.backend.model_id, {
            "support_consistency_is_heldout_accuracy": False,
            "query_images_used_in_proposal": False, "query_labels_used": False,
            "verification": "frozen concept, one image per prompt, no group label",
            "consistent_concepts": len(eligible), "local_only": True,
            "support_search_rounds": rounds, "maximum_rounds": self.max_rounds,
            "support_selected_measurement_axes": self._grounding.names if self._grounding else (),
            "limitations": "Same model proposes and checks; correlated errors and spurious concepts remain possible.",
        })

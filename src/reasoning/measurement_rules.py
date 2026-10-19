"""Executable numeric separators with support-only refitting and explicit units."""
from dataclasses import asdict, dataclass
from itertools import product
import numpy as np

from src.vision.observations import observations, DESCRIPTIONS
from .episode import EpisodePrediction
from .structural import _drawing


@dataclass(frozen=True)
class MeasurementRule:
    feature: str
    threshold: float
    positive_below: bool

    def assess(self, values):
        value = values.get(self.feature)
        if value is None:
            return None
        return bool(value < self.threshold) if self.positive_below else bool(value > self.threshold)

    @property
    def text(self):
        direction = "less than" if self.positive_below else "greater than"
        return f"The {DESCRIPTIONS[self.feature]} is {direction} {self.threshold:.3g}"


def _candidates(rows, labels, indices, features=None, all_candidates=False):
    candidates = []
    for name in DESCRIPTIONS if features is None else features:
        values = [rows[i][name] for i in indices]
        if any(value is None for value in values):
            continue
        p = np.array([rows[i][name] for i in indices if labels[i]])
        n = np.array([rows[i][name] for i in indices if not labels[i]])
        for below, low, high in ((True,p,n), (False,n,p)):
            gap = float(high.min()-low.max())
            span = float(max(high.max(),low.max())-min(high.min(),low.min()))
            if gap <= max(1e-8,span*.12):
                continue
            rule = MeasurementRule(name,float((high.min()+low.max())/2),below)
            candidates.append((gap/max(span,1e-8),rule))
    if not candidates:
        return []
    if all_candidates:
        return candidates
    best = max(score for score, rule in candidates)
    return [rule for score, rule in candidates if abs(score-best) < 1e-7]


def _vote(rules, values):
    votes = [rule.assess(values) for rule in rules]
    return votes[0] if votes and votes[0] is not None and all(v is votes[0] for v in votes) else None


@dataclass
class MeasurementFit:
    rules: tuple
    diagnostics: dict

    def predict(self, images, image_ids):
        result = []
        for image, image_id in zip(images,image_ids):
            values = observations(image) if _drawing(image) else {}
            label = _vote(self.rules,values)
            votes = [rule.assess(values) for rule in self.rules]
            result.append(EpisodePrediction(image_id,label,label,0. if label is None else 1. if label else -1.,None,
                () if label is not None else ("unresolved_measurement_or_program_disagreement",),(),
                {"rule_votes": votes,"measured_evidence": {rule.feature: values.get(rule.feature) for rule in self.rules}}))
        return result


def fit_measurement_rules(positive, negative, *, progress=None):
    images = list(positive)+list(negative)
    attempt = {"accepted":False, "query_images_used_in_selection":False}
    if min(len(positive),len(negative)) < 3 or max(len(positive),len(negative)) > 8 or not all(_drawing(i) for i in images):
        return None, {**attempt,"reason":"outside_measurement_rule_domain"}
    if progress:
        progress("Checking explicit geometric measurements with support-only refitting")
    rows = [observations(image) for image in images]
    labels = [True]*len(positive)+[False]*len(negative)
    indices = list(range(len(images)))
    candidates = _candidates(rows,labels,indices,all_candidates=True)
    if not candidates:
        return None, {**attempt,"reason":"no_separating_measurement"}
    eligible, validation = [], []
    for score, candidate in candidates:
        correct = total = 0
        for a,b in product(range(len(positive)),range(len(positive),len(images))):
            rules = _candidates(rows,labels,[i for i in indices if i not in (a,b)],features=(candidate.feature,))
            for i in (a,b):
                correct += _vote(rules,rows[i]) is labels[i]
                total += 1
        validation.append({"feature":candidate.feature,"correct":correct,"total":total})
        if correct == total:
            eligible.append((score,candidate))
    attempt.update(validation_total=total,validation_folds=total//2,
                   validation_method="threshold_refits_per_feature_then_support_selection",
                   validation_is_independent_accuracy=False,feature_validation=validation)
    if not eligible:
        return None,{**attempt,"reason":"measurement_support_validation_failed"}
    # Do not erase equally simple explanations just because one has a larger
    # observed margin. A query that separates those alternatives is unresolved.
    eligible.sort(key=lambda item:(-item[0],item[1].feature))
    chosen = [rule for score,rule in eligible]
    attempt.update(validation_correct=total,validation_accuracy=1.,validated_feature_count=len(eligible))
    attempt.update(accepted=True,reason="validated_measurement_rules",candidate_count=len(chosen),
        support_correct=len(images),support_total=len(images),
        candidates=[{"text":rule.text,"program":{"op":"measurement_threshold",**asdict(rule)}} for rule in chosen],
        support_evidence=rows)
    return MeasurementFit(tuple(chosen),attempt),attempt

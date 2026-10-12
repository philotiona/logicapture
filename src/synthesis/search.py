from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
import numpy as np
from src.dsl.grammar import enumerate_rules
from src.dsl.nodes import Rule, And, Or, Not
from src.dsl.predictions import prediction_matrix


@dataclass(frozen=True)
class RankedRule:
    rule: Rule
    correct: int
    total: int
    score: float
    unknown: int = 0

    @property
    def accuracy(self): return self.correct / self.total

    @property
    def coverage(self): return (self.total-self.unknown) / self.total

    @property
    def answered_accuracy(self):
        return self.correct / max(1, self.total-self.unknown)


@dataclass
class SearchResult:
    ranked: list[RankedRule]
    perfect_count: int
    evaluated: int
    seconds: float
    semantic_groups: int
    expanded: bool


def search(examples, depth=2, color=False, noisy=False, complexity_penalty=0.003,
           beam=40, limit=8, paths=False):
    if len(examples) < 2 or {bool(y) for _, y in examples} != {False, True}:
        raise ValueError("Provide at least one example in each group")
    started = perf_counter()
    labels = np.array([bool(y) for _, y in examples])
    rules = enumerate_rules(depth, color, paths=True) if paths else enumerate_rules(depth, color)
    needs_assessment = paths or any(obj.kind != "region" or obj.shape == "unknown"
                                    for scene, _ in examples for obj in scene.objects)
    if needs_assessment:
        from src.dsl.predictions import assessment_matrix
        matrix = assessment_matrix(rules, [s for s, _ in examples])
    else:
        matrix = prediction_matrix(rules, [s for s, _ in examples]).astype(np.int8)
    predictions = dict(zip(rules, matrix))
    expanded = False
    def size(rule):
        # A numeric constant adds a constraint. Prefer an existential direction
        # relationship over 'exactly one such path' when both fit identically.
        constants = int(rule.op in ("exactly", "atleast", "compare"))
        constants += sum(size(c)-c.complexity for c in rule.children)
        return rule.complexity + (constants if paths else 0)
    def order(rule):
        errors = int(np.count_nonzero(predictions[rule] != labels))
        unknown = int(np.count_nonzero(predictions[rule] < 0))
        return errors, unknown, size(rule), rule.key
    if not any(np.array_equal(v, labels) for v in predictions.values()) and depth >= 2:
        # Expand promising partial rules; keep multiple observationally equivalent
        # structures for ambiguity reporting, but only one per beam signature.
        seen, seeds = set(), []
        for r in sorted(rules, key=order):
            signature = predictions[r].tobytes()
            if signature not in seen:
                seen.add(signature)
                seeds.append(r)
            if len(seeds) >= beam: break
        for a, b in combinations(seeds, 2):
            av, bv = predictions[a], predictions[b]
            conjunction = np.where((av == 0) | (bv == 0), 0,
                                   np.where((av == 1) & (bv == 1), 1, -1)).astype(np.int8)
            disjunction = np.where((av == 1) | (bv == 1), 1,
                                   np.where((av == 0) & (bv == 0), 0, -1)).astype(np.int8)
            for r, v in ((And(a, b), conjunction), (Or(a, b), disjunction)):
                if r.depth <= depth: predictions[r] = v
        expanded = True
    ranked = []
    perfect = 0
    signatures = set()
    for rule, v in predictions.items():
        correct = int(np.count_nonzero(v == labels))
        perfect += correct == len(labels)
        signatures.add(v.tobytes())
        unknown = int(np.count_nonzero(v < 0))
        score = correct/len(labels) - complexity_penalty*size(rule) - .05*unknown/len(labels)
        ranked.append(RankedRule(rule, correct, len(labels), score, unknown))
    ranked.sort(key=(lambda r: (-r.score, r.unknown, size(r.rule), r.rule.key)) if noisy else
                (lambda r: (-r.correct, r.unknown, size(r.rule), r.rule.key)))
    return SearchResult(ranked[:limit], perfect, len(predictions), perf_counter()-started, len(signatures), expanded)

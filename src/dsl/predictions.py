"""Cached hard evaluation across a dataset; equivalent to Rule.evaluate.

Count queries share selector counts, and logical programs reuse child vectors.
This permits deliberate rare-concept coverage without slow per-AST repetition.
"""
from itertools import combinations
import numpy as np


def prediction_matrix(rules, scenes):
    if any(obj.kind != "region" or obj.shape == "unknown" or obj.path is not None for scene in scenes for obj in scene.objects):
        return assessment_matrix(rules, scenes) == 1
    objects = [scene.objects for scene in scenes]
    totals = np.array([len(group) for group in objects])
    selections, counts, values = {}, {}, {}

    def selected(predicate):
        if predicate not in selections:
            selections[predicate] = [tuple(obj.id for obj in group if predicate(obj)) for group in objects]
            counts[predicate] = np.array([len(ids) for ids in selections[predicate]])
        return selections[predicate]

    def evaluate(rule):
        if rule in values: return values[rule]
        op = rule.op
        if op in ("and", "or", "implies", "not"):
            a = evaluate(rule.children[0])
            if op == "not": value = ~a
            else:
                b = evaluate(rule.children[1])
                value = a & b if op == "and" else a | b if op == "or" else ~a | b
        else:
            left = selected(rule.predicates[0])
            n = counts[rule.predicates[0]]
            if op == "exists": value = n > 0
            elif op == "forall": value = n == totals
            elif op == "exactly": value = n == rule.n
            elif op == "atleast": value = n >= rule.n
            elif op == "compare":
                selected(rule.predicates[1])
                right = counts[rule.predicates[1]]
                value = {"gt": n > right, "lt": n < right, "eq": n == right, "approx": np.abs(n-right) <= 1}[rule.relation]
            elif op == "aligned":
                value = np.array([any(("aligned", *sorted(ids)) in scene.relations for ids in combinations(group, 3))
                                  for scene, group in zip(scenes, left)], bool)
            else:
                right = selected(rule.predicates[1])
                reducer = any if op == "pair" else all
                value = np.array([reducer(any(a != b and (rule.relation, a, b) in scene.relations for b in rhs)
                                         for a in lhs) for scene, lhs, rhs in zip(scenes, left, right)], bool)
        values[rule] = np.asarray(value, dtype=bool)
        return values[rule]

    return np.stack([evaluate(rule) for rule in rules]) if rules else np.empty((0, len(scenes)), dtype=bool)


def assessment_matrix(rules, scenes):
    """int8 evidence matrix: -1 unavailable, 0 disproved, 1 supported.

The existing boolean evaluator is retained for generated fully known scenes.
Search must not coerce this matrix to bool: -1 is unavailable evidence.
"""
    return np.array([[(-1 if (value := rule.assess(scene)) is None else int(value))
                      for scene in scenes] for rule in rules], dtype=np.int8).reshape(len(rules), len(scenes))

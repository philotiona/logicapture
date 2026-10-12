from dataclasses import dataclass
import numpy as np
from src.dsl.grammar import enumerate_rules
from src.dsl.predictions import prediction_matrix
from .scene_generation import SceneGenerator
from .query_selection import select_query, select_query_batch


@dataclass
class Answer:
    query_index: int
    label: bool
    survivors: int
    query_indices: tuple[int, ...] = ()


class AkinatorSession:
    revision = 2

    def __init__(self, seed=42, color=False, depth=2, pool_size=280, strategy="active", rules=None, scenes=None):
        self.rules = list(rules) if rules is not None else enumerate_rules(depth, color)
        self.color = color
        self.scenes = list(scenes) if scenes is not None else SceneGenerator(seed, color).pool(pool_size)
        self.predictions = prediction_matrix(self.rules, self.scenes)
        # Equal treatment of primitive counted combinations: adding a fill
        # selector should not make a simple count rule almost invisible.
        def prior(rule):
            if rule.op in ("exactly", "atleast", "exists", "forall"): return 1.0
            if rule.op == "and" and any(c.op in ("exactly", "atleast") and c.predicates[0].op == "any" for c in rule.children) and any(c.op == "forall" for c in rule.children):
                return 1.0
            return .1 if rule.op not in ("and", "or", "not") else .005
        self.rule_weights = np.array([prior(r) for r in self.rules])
        self.survivors = np.arange(len(self.rules))
        self.history = []
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)
        self.current = self._next()

    def _next(self):
        # A positive union answer does not label its individual drawings. Keep
        # those drawings available for follow-up questions to resolve the match.
        answered = {index for a in self.history if not a.label or len(a.query_indices or (a.query_index,)) == 1
                    for index in (a.query_indices or (a.query_index,))}
        available = np.array([i for i in range(len(self.scenes)) if i not in answered], dtype=int)
        weights = self.rule_weights[self.survivors] if hasattr(self, "rule_weights") else None
        first = select_query(self.predictions, self.survivors, available, self.strategy, self.rng, weights)
        self.query_indices = () if first is None else (first,)
        if first is not None and self.strategy == "active" and len(self.history) >= 6 and not any(a.label for a in self.history):
            self.query_indices = select_query_batch(self.predictions, self.survivors, available, first, weights)
        return first

    @property
    def candidates(self):
        return [self.rules[i] for i in self.survivors]

    @property
    def scene(self): return self.scenes[self.current] if self.current is not None else None

    @property
    def query_scenes(self): return [self.scenes[i] for i in self.query_indices]

    def evaluate_query(self, rule):
        return any(rule.evaluate(scene) for scene in self.query_scenes)

    @property
    def equivalence_count(self):
        return len({self.predictions[i].tobytes() for i in self.survivors})

    def answer(self, label):
        if self.current is None: raise ValueError("There is no pending question")
        predictions = self.predictions[np.ix_(self.survivors, self.query_indices)].any(axis=1)
        self.survivors = self.survivors[predictions == bool(label)]
        self.history.append(Answer(self.current, bool(label), len(self.survivors), self.query_indices))
        self.current = self._next()

    def undo(self):
        if not self.history: return
        last = self.history.pop()
        self.survivors = np.arange(len(self.rules))
        for answer in self.history:
            indices = answer.query_indices or (answer.query_index,)
            predictions = self.predictions[np.ix_(self.survivors, indices)].any(axis=1)
            self.survivors = self.survivors[predictions == answer.label]
        self.current = last.query_index
        self.query_indices = last.query_indices or (last.query_index,)

    def add_query_pool(self, seed, count=120, color=False):
        extra = SceneGenerator(seed, color).pool(count)
        values = prediction_matrix(self.rules, extra)
        self.scenes.extend(extra)
        self.predictions = np.concatenate((self.predictions, values), axis=1)
        self.current = self._next()

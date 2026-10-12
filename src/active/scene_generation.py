from random import Random
from dataclasses import replace
from itertools import product
from math import ceil
from src.scene import VisualObject, Scene, derive_relations


class GenerationError(RuntimeError):
    pass


class SceneGenerator:
    def __init__(self, seed=42, color=False):
        self.rng = Random(seed)
        self.color = color

    def random_scene(self, count=None, nested=None):
        count = self.rng.randint(0, 6) if count is None else count
        if not 0 <= count <= 8:
            raise ValueError("Generator supports 0–8 objects")
        nested = (self.rng.random() < .32 and count == 2) if nested is None else nested
        objects = []
        slots = [(x, y) for y in (.2, .5, .8) for x in (.2, .5, .8)]
        self.rng.shuffle(slots)
        for i in range(count):
            shape = self.rng.choice(("circle", "triangle", "quadrilateral"))
            area = self.rng.uniform(.008 if count >= 5 else .012, .045 if count <= 2 else .014 if count >= 5 else .018)
            x, y = .5, .5
            if count >= 3:
                x, y = slots[i]
                x += self.rng.uniform(-.025, .025)
                y += self.rng.uniform(-.025, .025)
            else:
                for _ in range(150):
                    x, y = self.rng.uniform(.19, .81), self.rng.uniform(.19, .81)
                    if all((x-o.center_x)**2+(y-o.center_y)**2 > .10 for o in objects):
                        break
            objects.append(VisualObject(i, shape, self.rng.choice((True, False)),
                                        self.rng.choice(("black", "red", "blue", "green")) if self.color else "black",
                                        x, y, area, self.rng.uniform(-25, 25)))
        if nested and count >= 2:
            # Distinct outline container and a comfortably smaller enclosed object.
            outer_shape = self.rng.choice(("circle", "quadrilateral"))
            objects[0] = VisualObject(0, outer_shape, False, "black", .5, .5, .24)
            objects[1] = VisualObject(1, self.rng.choice(("circle", "triangle", "quadrilateral")),
                                      self.rng.choice((True, False)), "black", .5, .5, .015)
        return derive_relations(Scene(objects))

    def predicate_scene(self, predicate, matches, distractors=0):
        """Construct selector counts, independent of any complete hidden rule.

        Distractors explicitly separate counts of matching objects from total
        counts. Grid packing keeps high-count queries readable without overlap.
        """
        total = matches+distractors
        if not 0 <= total <= 8:
            raise GenerationError("Constructive scenes support at most eight objects")
        area = .012 if total >= 5 else .025
        candidates = [VisualObject(0, shape, filled, color, area=area)
                      for shape, filled, color in product(
                          ("circle", "triangle", "quadrilateral"), (True, False),
                          ("black", "red", "blue", "green") if self.color else ("black",))]
        yes = [o for o in candidates if predicate(o)]
        no = [o for o in candidates if not predicate(o)]
        if matches and not yes or distractors and not no:
            raise GenerationError("Selector has no constructive objects in this domain")
        objects = [self.rng.choice(yes) for _ in range(matches)] + [self.rng.choice(no) for _ in range(distractors)]
        return self._pack(objects)

    def _pack(self, objects):
        total = len(objects)
        self.rng.shuffle(objects)
        cols = 1 if total == 1 else 2 if total <= 4 else 3
        rows = max(1, ceil(total/cols))
        slots = [((col+.5)/cols, (row+.5)/rows) for row in range(rows) for col in range(cols)]
        self.rng.shuffle(slots)
        packed = [replace(obj, id=i, center_x=x+self.rng.uniform(-.025, .025),
                          center_y=y+self.rng.uniform(-.025, .025), angle=self.rng.uniform(-25, 25))
                  for i, (obj, (x, y)) in enumerate(zip(objects, slots))]
        return derive_relations(Scene(packed))

    def _guided(self, rule, label, count=None):
        if rule.op == "and":
            counter = next((c for c in rule.children if c.op in ("exactly", "atleast") and c.predicates[0].op == "any"), None)
            every = next((c for c in rule.children if c.op == "forall"), None)
            if counter and every and 0 <= counter.n <= 8:
                try:
                    scene = self.predicate_scene(every.predicates[0], counter.n if label else max(0, counter.n-1), 0 if label else 1)
                    if rule.evaluate(scene) == bool(label) and (count is None or len(scene.objects) == count): return scene
                except GenerationError: pass
            return None
        if rule.op not in ("exactly", "atleast", "exists", "forall"):
            return None
        predicate = rule.predicates[0]
        if rule.op in ("exactly", "atleast"):
            matches = rule.n if label else (rule.n-1 if rule.n else 1)
            if label and rule.op == "atleast": matches = min(8, matches+self.rng.randint(0, 1))
        elif rule.op == "exists":
            matches = self.rng.randint(1, 3) if label else 0
        else:
            matches = self.rng.randint(1, 4)
        if not 0 <= matches <= 8 or count is not None and matches > count: return None
        distractors = count-matches if count is not None else (1 if rule.op == "forall" and not label else 0)
        if rule.op == "forall" and label: distractors = 0
        try:
            scene = self.predicate_scene(predicate, matches, distractors)
            return scene if rule.evaluate(scene) == bool(label) and (count is None or len(scene.objects) == count) else None
        except GenerationError:
            return None

    def sample(self, rule, label, max_attempts=4000, count=None):
        # Rare counted combinations should not depend on winning a lottery.
        guided = self._guided(rule, bool(label), count)
        if guided is not None: return guided
        for _ in range(max_attempts):
            scene = self.random_scene(count=count)
            if rule.evaluate(scene) == bool(label):
                return scene
        raise GenerationError(f"Could not generate a {'positive' if label else 'negative'} example in {max_attempts} attempts. The rule may be rare or constant in the generator domain.")

    def balanced_examples(self, rule, per_group=6):
        if rule.op in ("exactly", "atleast") and rule.predicates[0].op == "and" and 1 <= rule.n <= 6:
            selector = rule.predicates[0]
            positives = [(self.predicate_scene(selector, rule.n, i%3), True) for i in range(per_group)]
            negatives = []
            for i in range(per_group):
                matches = rule.n+1 if rule.op == "exactly" and i%3 == 1 else max(0, rule.n-1)
                distractors = 0 if matches > rule.n else 1+i%2
                negatives.append((self.predicate_scene(selector, matches, distractors), False))
            return positives+negatives
        if rule.op == "and":
            from src.dsl.nodes import And, Not
            positives = [(self.sample(rule, True), True) for _ in range(per_group)]
            negatives = []
            for i in range(per_group):
                a, b = rule.children if i % 2 == 0 else tuple(reversed(rule.children))
                try:
                    # Violating each clause in turn makes each clause necessary.
                    scene = self.sample(And(Not(a), b), True, max_attempts=1000)
                except GenerationError:
                    scene = self.sample(rule, False)
                negatives.append((scene, False))
            return positives + negatives
        # For the basic presence families, use exactly the same counts in both
        # groups. Otherwise "more objects" can be a synthetic shortcut.
        if rule.op == "exists" and rule.predicates[0].op in ("shape", "filled", "outline", "color"):
            counts = [1+i%4 for i in range(per_group)]
            return [(self.sample(rule, label, count=count), label)
                    for label in (True, False) for count in counts]
        return [(self.sample(rule, label), label) for label in (True, False) for _ in range(per_group)]

    def pool(self, size=280):
        from src.dsl.grammar import counted_predicates
        from itertools import combinations
        scenes = [self.random_scene(count=i % 7) for i in range(size)]
        for selector in counted_predicates(self.color):
            for count in range(8):
                for distractors in (0, 1):
                    if count+distractors > 8: continue
                    try: scenes.append(self.predicate_scene(selector, count, distractors))
                    except GenerationError: pass
        # Mixed counts test several attribute families in a single drawing and
        # separate a matching-object count from the total number of objects.
        atoms = [p for p in counted_predicates(False) if p.op == "and"]
        for a, b in combinations(atoms, 2):
            for na, nb in ((2, 2), (3, 3), (4, 4), (4, 2), (2, 4)):
                first, second = self.predicate_scene(a, na), self.predicate_scene(b, nb)
                scenes.append(self._pack(first.objects+second.objects))
        return scenes

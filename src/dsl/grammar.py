from itertools import combinations, product
from .nodes import Predicate, Rule, Exists, ForAll, Count, Relation, Compare, And, Or, Not


def primitive_predicates(color=False):
    result = [Predicate(), *[Predicate("shape", s) for s in ("circle", "triangle", "quadrilateral")],
              Predicate("filled"), Predicate("outline"), Predicate("small"), Predicate("large")]
    if color:
        result += [Predicate("color", c) for c in ("black", "red", "blue", "green")]
    return result


def counted_predicates(color=False):
    """Reusable selectors for counts and for systematic query coverage."""
    p = primitive_predicates(color)
    selectors = p[:6] + (p[8:] if color else [])
    selectors += [Predicate("and", children=(shape, prop))
                  for shape in p[1:4] for prop in p[4:6]]
    if color:
        selectors += [Predicate("and", children=(shape, prop))
                      for shape in p[1:4] for prop in p[8:]]
    return selectors


def path_predicates():
    """Reusable graph features, independent of any complete Bongard rule."""
    return [Predicate("kind", "path"),
            *[Predicate("topology", topology) for topology in ("open", "closed", "branched")],
            *[Predicate("geometry", geometry) for geometry in ("straight", "curved")],
            Predicate("has_arrows"),
            *[Predicate("arrow_relation", relation) for relation in ("same", "opposed")],
            *[Predicate("arrow_count", str(count)) for count in (0, 1, 2, 3)],
            *[Predicate("endpoint_count", str(count)) for count in (0,2,3,4)],
            *[Predicate("junction_count", str(count)) for count in (0,1,2)],
            *[Predicate("cycle_count", str(count)) for count in (0,1,2)]]


def enumerate_rules(depth=2, color=False, max_count=6, paths=False):
    """Bounded, reproducible grammar; primitives compose, no whole-rule dictionary.

    Depth means scene-rule composition depth. Object selector depth is bounded
    separately at two. For full dataset-driven composition use synthesis.search.
    """
    if depth not in (1, 2, 3):
        raise ValueError("Supported search depths: 1–3")
    if not 1 <= max_count <= 7:
        raise ValueError("Supported count bound: 1–7")
    p = primitive_predicates(color)
    rules = []
    for predicate in p:
        rules.extend([Exists(predicate), ForAll(predicate)])
        # Large-object counts need a separate layout domain; keep size as a
        # primitive without promising high counts of overlapping large shapes.
        count_bound = max_count if predicate.op not in ("small", "large") else 3
        rules.extend(Count(n, predicate) for n in range(count_bound+1))
        rules.extend(Count(n, predicate, at_least=True) for n in range(2, count_bound+1))
    for shape in p[1:4]:
        for prop in p[4:6]:
            conjunction = Predicate("and", children=(shape, prop))
            implication = Predicate("or", children=(Predicate("not", children=(shape,)), prop))
            rules.extend([Exists(conjunction), ForAll(conjunction), ForAll(implication)])
    for selector in counted_predicates(color):
        rules.extend(Count(n, selector) for n in range(max_count+1))
        rules.extend(Count(n, selector, at_least=True) for n in range(2, max_count+1))
    pair_selectors = p[:6]
    for relation in ("left_of", "above", "inside", "near", "larger_than", "same_shape", "touching", "overlapping", "same_size"):
        for a, b in product(pair_selectors, repeat=2):
            rules.append(Relation(relation, a, b))
            if relation == "inside": rules.append(Relation(relation, a, b, universal=True))
    for a, b in combinations(p[:6], 2):
        rules.extend(Compare(a, b, op) for op in ("gt", "eq", "approx"))
    rules.append(Rule("aligned", (Predicate(),)))
    if color:
        rules.extend(Relation("same_color", a, b) for a, b in product(p[1:4], repeat=2))
    base = list(dict.fromkeys(rules))
    if depth >= 2:
        rules.extend(Not(r) for r in base)
        # A deliberate finite vocabulary for interactive questioning.
        seeds = [Exists(q) for q in p] + [Count(n) for n in range(1, max_count+1)]
        seeds += [ForAll(q) for q in counted_predicates(color)[1:]]
        seeds += [Count(2, q) for q in p[1:4]]
        seeds += [Relation(r, a, b) for r in ("left_of", "above", "inside") for a, b in product(p[1:4], repeat=2)]
        for a, b in combinations(seeds, 2):
            rules.extend((And(a, b), Or(a, b)))
    if depth >= 3:
        seeds = [Exists(q) for q in p[:6]]
        rules.extend(And(And(a, b), c) for a, b, c in combinations(seeds, 3))
    if paths:
        selectors = path_predicates()
        # Region and graph vocabularies share the same executable quantifiers.
        graph_rules = [constructor(selector) for selector in selectors for constructor in (Exists, ForAll)]
        graph_rules += [Count(n, selector) for selector in selectors for n in range(min(max_count, 3)+1)]
        graph_rules += [Count(2, selector, at_least=True) for selector in selectors]
        composed = [Predicate("and", children=(topology, feature))
                    for topology in selectors[1:4] for feature in selectors[4:9]]
        graph_rules += [constructor(selector) for selector in composed for constructor in (Exists, ForAll)]
        rules.extend(graph_rules)
        if depth >= 2:
            rules.extend(Not(rule) for rule in graph_rules)
            seeds = [ForAll(selector) for selector in selectors[1:9]]
            seeds += [Count(n, Predicate("kind", "path")) for n in (1, 2, 3)]
            for a, b in combinations(seeds, 2): rules.extend((And(a, b), Or(a, b)))
    return sorted(set(rules), key=lambda r: (r.complexity, r.key))

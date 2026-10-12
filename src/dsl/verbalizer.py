from .nodes import Rule


def describe_selector(predicate, plural=False):
    """Readable noun phrases for reusable, composed object predicates."""
    parts = []
    def flatten(p):
        if p.op == "and":
            for child in p.children: flatten(child)
        else: parts.append(p)
    flatten(predicate)
    if all(p.op in ("kind", "topology", "geometry", "has_arrows", "arrow_relation", "arrow_count",
                   "endpoint_count", "junction_count", "cycle_count") for p in parts):
        noun = "paths" if plural else "path"
        modifiers = [p.value for p in parts if p.op in ("topology", "geometry")]
        phrase = " ".join(modifiers+[noun])
        for p in parts:
            if p.op == "kind" and p.value != "path": phrase = p.value+("s" if plural else "")
            if p.op == "has_arrows": phrase += " with arrows"
            if p.op == "arrow_count": phrase += f" with {p.value} arrow"+("s" if p.value != "1" else "")
            if p.op in ("endpoint_count","junction_count","cycle_count"):
                phrase += f" with {p.value} {p.op.removesuffix('_count')}"+("s" if p.value!="1" else "")
            if p.op == "arrow_relation":
                phrase += " whose arrows point "+("the same way" if p.value == "same" else "opposite ways")+" along the path"
        return phrase
    if all(p.op in ("any", "shape", "color", "filled", "outline", "small", "large") for p in parts):
        noun = next((p.value for p in parts if p.op == "shape"), "object")
        modifiers = [p.value if p.op == "color" else p.op for p in parts if p.op not in ("any", "shape")]
        return " ".join(modifiers+[noun+("s" if plural else "")])
    return f"objects matching {predicate.text}" if plural else f"object matching {predicate.text}"


def article(text):
    return ("an " if text[0].lower() in "aeiou" else "a ")+text


def verbalize(rule: Rule):
    op = rule.op
    if op == "not": return f"Not ({verbalize(rule.children[0])})"
    if op in ("and", "or", "implies"):
        if op == "and":
            count = next((c for c in rule.children if c.op in ("exactly", "atleast") and c.predicates[0].op == "any"), None)
            every = next((c for c in rule.children if c.op == "forall"), None)
            if count and every:
                return f"{verbalize(count)}, all {describe_selector(every.predicates[0], True)}"
        join = {"and": "and", "or": "or", "implies": "implies"}[op]
        return f"({verbalize(rule.children[0])}) {join} ({verbalize(rule.children[1])})"
    p = describe_selector(rule.predicates[0])
    plural = describe_selector(rule.predicates[0], plural=True)
    if op == "exists": return f"There is at least one {p}"
    if op == "forall": return f"Every object is {article(p)}"
    if op in ("exactly", "atleast"):
        quant = "exactly" if op == "exactly" else "at least"
        if rule.n == 0 and op == "exactly": return f"There are no {plural}"
        if rule.n == 1: return f"There is {quant} one {p}"
        return f"There are {quant} {rule.n} {plural}"
    if op == "aligned": return f"At least three {plural} are approximately collinear"
    q = describe_selector(rule.predicates[1])
    if op == "compare":
        words = {"gt": "greater than", "lt": "less than", "eq": "equal to", "approx": "within one of"}
        return f"The number of {plural} is {words[rule.relation]} the number of {describe_selector(rule.predicates[1], True)}"
    rel = {"left_of": "left of", "right_of": "right of", "above": "above", "below": "below",
           "inside": "inside", "touching": "touching", "overlapping": "overlapping",
           "near": "near", "larger_than": "larger than", "same_shape": "the same shape as",
           "same_color": "the same color as", "same_size": "approximately the same size as"}[rule.relation]
    prefix = f"Every {p} is" if op == "forall_pair" else f"At least one {p} is"
    return f"{prefix} {rel} {article(q)}"

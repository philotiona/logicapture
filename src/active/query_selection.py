import numpy as np


def select_query_batch(predictions, survivor_indices, available, first, rule_weights=None, maximum=4):
    """Greedy union query: user labels whether any independent drawing fits.

    This avoids a long sequence of singleton negative tests for highly specific
    concepts, while keeping exactly executable version-space updates.
    """
    behaviors, groups = np.unique(predictions[survivor_indices], axis=0, return_inverse=True)
    weights = np.ones(len(behaviors))
    if rule_weights is not None:
        weights[:] = 0
        np.maximum.at(weights, groups, rule_weights)
        constant = (behaviors.sum(axis=1) == 0) | (behaviors.sum(axis=1) == behaviors.shape[1])
        weights[constant] = 0
        if not weights.any(): weights[:] = 1
    chosen, union = [first], behaviors[:, first].copy()
    score = abs(2*(weights @ union)-weights.sum())
    for _ in range(maximum-1):
        remaining = np.array([i for i in available if i not in chosen])
        if not len(remaining): break
        unions = union[:, None] | behaviors[:, remaining]
        scores = np.abs(2*(weights @ unions)-weights.sum())
        best = int(np.argmin(scores))
        if scores[best] >= score-1e-9: break
        chosen.append(int(remaining[best]))
        union, score = unions[:, best], float(scores[best])
    return tuple(chosen)


def select_query(predictions, survivor_indices, available, strategy="active", rng=None, rule_weights=None):
    if strategy not in ("active", "random"):
        raise ValueError("Unknown query strategy")
    if not len(survivor_indices) or not len(available): return None
    # Syntax duplicates must not receive extra votes simply because many ways
    # to spell AND/NOT happen to express the same observable behavior.
    behaviors, groups = np.unique(predictions[survivor_indices], axis=0, return_inverse=True)
    yes = behaviors[:, available].sum(axis=0)
    useful = (yes > 0) & (yes < len(behaviors))
    if not useful.any(): return None
    if strategy == "random":
        # Passive baseline samples the same candidate pool uniformly, including
        # uninformative scenes. Active selection alone uses disagreement.
        return int(rng.choice(available))
    if rule_weights is None:
        balance = np.abs(2*yes-len(behaviors)).astype(float)
    else:
        weights = np.zeros(len(behaviors))
        np.maximum.at(weights, groups, np.asarray(rule_weights))
        # A rule that is constant on the entire pool provides no experiment of
        # its own. It must not dominate the prior and postpone rare witnesses.
        constant = (behaviors.sum(axis=1) == 0) | (behaviors.sum(axis=1) == behaviors.shape[1])
        weights[constant] = 0
        if not weights.any(): weights[:] = 1
        balance = np.abs(2*(weights @ behaviors[:, available])-weights.sum())
    balance[~useful] = np.inf
    return int(available[np.argmin(balance)])

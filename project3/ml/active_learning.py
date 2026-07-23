"""
Task 4: Active learning for expert-competence discovery.

Setting (per the assignment): we have the whole training set for the AI
classifier (already trained in Task 1), but we do NOT have any expert
labels up front. We may query the simulated expert one example at a time,
and each query has a cost, so the goal is to learn -- with as few queries
as possible -- when deferring to the expert is actually worthwhile.

This re-uses the Task 3 deferral gate, but trains it only on the small set
of points we actively chose to query, instead of on a large pre-labelled
slice.

Acquisition strategies compared
--------------------------------
- random:       query uniformly random points (control / lower bound)
- uncertainty:  query the AI's least-confident points first
                (lowest top-1 predicted probability)
- margin:       query points with the smallest gap between the AI's top-1
                and top-2 predicted class probabilities

Uncertainty/margin sampling are natural choices here because the deferral
decision is fundamentally about the AI's own confidence: the gate can only
learn "the AI is unreliable AND the expert is reliable HERE" from points
where the AI's confidence signal is informative in the first place.
Querying only confident points would rarely reveal AI mistakes to learn
from, while querying near the AI's decision boundary concentrates the
(scarce) expert-labelling budget exactly where the deferral decision is
hardest -- and least confident.

For a series of increasing query budgets we: (1) pick that many points via
each strategy, (2) query the simulated expert on exactly those points,
(3) train a deferral gate using only that labelled subset, (4) evaluate
the resulting learning-to-defer system on the held-out test set. This
produces a learning curve of system test accuracy vs. number of expert
queries, per strategy.
"""

import numpy as np

from .deferral import DeferralGate, evaluate_system


def select_indices(strategy, probs, budget, rng, already_picked=None):
    n = probs.shape[0]
    available = np.ones(n, dtype=bool)
    if already_picked is not None:
        available[already_picked] = False
    avail_idx = np.where(available)[0]
    budget = min(budget, len(avail_idx))

    if strategy == "random":
        return rng.choice(avail_idx, size=budget, replace=False)

    if strategy == "hybrid":
        # Half random, half uncertainty. Pure uncertainty/margin sampling
        # can starve the deferral gate of easy, AI-correct examples at very
        # small budgets, producing a degenerate "always defer" gate (see
        # report). Mixing in random exploration keeps both classes
        # ("deferring helps" / "deferring doesn't help") represented.
        n_random = budget // 2
        n_uncertain = budget - n_random
        random_part = rng.choice(avail_idx, size=n_random, replace=False)
        remaining = np.setdiff1d(avail_idx, random_part, assume_unique=False)
        sorted_probs = np.sort(probs[remaining], axis=1)
        score = sorted_probs[:, -1]
        order = np.argsort(score)[:n_uncertain]
        return np.concatenate([random_part, remaining[order]])

    sorted_probs = np.sort(probs[avail_idx], axis=1)
    if strategy == "uncertainty":
        score = sorted_probs[:, -1]  # lower top-1 confidence = more uncertain
    elif strategy == "margin":
        score = sorted_probs[:, -1] - sorted_probs[:, -2]  # smaller margin = more ambiguous
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    order = np.argsort(score)  # ascending: least confident / smallest margin first
    chosen_local = order[:budget]
    return avail_idx[chosen_local]


def run_budget_curve(
    strategy,
    train_probs,
    train_ai_preds,
    train_true_labels,
    expert,
    test_probs,
    test_ai_preds,
    test_true_labels,
    test_expert_preds,
    budgets,
    seed=0,
):
    """
    Runs one acquisition strategy across a list of increasing query budgets
    and returns a list of {budget, metrics} entries (a learning curve).
    """
    rng = np.random.default_rng(seed)
    curve = []
    for budget in budgets:
        idx = select_indices(strategy, train_probs, budget, rng)

        # Split the queried budget into a fit slice and a small calibration
        # slice (used to pick the deferral threshold -- see deferral.py).
        # Below ~20 queries there isn't enough data to calibrate reliably,
        # so we fall back to fitting on everything with the default 0.5 cut.
        gate = DeferralGate()
        if len(idx) >= 20:
            split = int(len(idx) * 0.7)
            fit_idx, cal_idx = idx[:split], idx[split:]
        else:
            fit_idx, cal_idx = idx, None

        fit_true = np.asarray(train_true_labels)[fit_idx]
        fit_expert_preds = expert.predict(fit_true, np.random.default_rng(seed + budget))
        gate.fit(train_probs[fit_idx], np.asarray(train_ai_preds)[fit_idx], fit_true, fit_expert_preds)

        if cal_idx is not None and len(cal_idx) >= 5:
            cal_true = np.asarray(train_true_labels)[cal_idx]
            cal_expert_preds = expert.predict(cal_true, np.random.default_rng(seed + budget + 1))
            gate.calibrate_threshold(train_probs[cal_idx], np.asarray(train_ai_preds)[cal_idx],
                                      cal_true, cal_expert_preds)

        defer_mask = gate.decide(test_probs)
        metrics = evaluate_system(test_ai_preds, test_expert_preds, test_true_labels, defer_mask)
        metrics["n_queries"] = int(len(idx))
        curve.append(metrics)
    return curve

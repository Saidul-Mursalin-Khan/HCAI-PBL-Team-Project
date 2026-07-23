"""
Task 3: Learning-to-defer.

Setting: both the AI classifier (Task 1) and expert labels (Task 2,
simulated) are available for training.

We need a deferral policy r(x) in {defer, predict} that decides, for each
input, whether the system should trust its own classifier or hand the
decision to the expert.

Design choice -- a LEARNED gate rather than a fixed confidence threshold
--------------------------------------------------------------------------
A simple baseline is "defer whenever the classifier's top-1 probability is
below some threshold". We implement that too (as a comparison point), but
the main system instead trains a small logistic-regression "gate" on a
held-out labelled slice of the training data. For every example in that
slice we compute:
  - the AI classifier's confidence (max softmax prob), margin (prob of the
    top class minus the second class) and entropy (uncertainty) over its
    predicted distribution,
  - whether the AI's own prediction was correct,
  - the simulated expert's prediction and whether IT was correct.

The gate's training target is:
    defer = 1  if the AI is wrong AND the expert would have been right
    defer = 0  otherwise (AI is right, or AI and expert are both wrong)

i.e. we only want to defer when doing so actually helps the outcome. This
directly optimises for the accuracy of the *combined* human-AI system,
which is what Task 3 asks us to evaluate, rather than only for confidence
calibration of the AI on its own.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def _features_from_probs(probs):
    """probs: (n, n_classes) predicted class probabilities from the AI classifier."""
    probs = np.clip(probs, 1e-9, 1.0)
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    margin = top1 - top2
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return np.column_stack([top1, margin, entropy])


class DeferralGate:
    """Learned rejector: predicts P(deferring helps) from AI-confidence features."""

    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.fitted_ = False

    def fit(self, probs, ai_preds, true_labels, expert_preds):
        X = _features_from_probs(probs)
        ai_correct = ai_preds == true_labels
        expert_correct = expert_preds == true_labels
        y = (~ai_correct & expert_correct).astype(int)
        # Edge case: if the target is single-class (e.g. tiny query budget in
        # Task 4), fall back to a trivial "never defer" gate instead of
        # crashing, so active-learning budget curves stay well defined.
        if len(np.unique(y)) < 2:
            self.fitted_ = "trivial"
            self._trivial_value = int(y[0]) if len(y) else 0
            return self
        self.clf.fit(X, y)
        self.fitted_ = True
        return self

    def defer_probability(self, probs):
        X = _features_from_probs(probs)
        if self.fitted_ == "trivial":
            return np.full(len(X), float(self._trivial_value))
        return self.clf.predict_proba(X)[:, 1]

    def decide(self, probs, threshold=None):
        t = threshold if threshold is not None else getattr(self, "threshold_", 0.5)
        return self.defer_probability(probs) >= t

    def calibrate_threshold(self, cal_probs, cal_ai_preds, cal_true_labels, cal_expert_preds,
                             thresholds=None):
        """
        The gate's raw defer_probability >= 0.5 rule is a poor default here:
        "deferring helps" is a RARE class (it only fires when the AI is wrong
        AND the expert happens to be right), so a naive 0.5 cutoff over-defers
        and can actually hurt system accuracy versus AI-only. Instead we pick
        the threshold, on a held-out calibration slice, that directly
        maximises combined system accuracy -- optimising for the metric we
        actually care about rather than for the gate's own classification
        accuracy.
        """
        thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
        probs_cal = self.defer_probability(cal_probs)
        best_t, best_acc = 0.5, -1.0
        for t in thresholds:
            mask = probs_cal >= t
            preds = np.where(mask, cal_expert_preds, cal_ai_preds)
            acc = float((preds == cal_true_labels).mean())
            if acc > best_acc:
                best_acc, best_t = acc, t
        self.threshold_ = float(best_t)
        return self.threshold_


def threshold_gate_decide(probs, confidence_threshold=0.7):
    """Simple baseline: defer whenever top-1 confidence is below a fixed threshold."""
    top1 = np.max(probs, axis=1)
    return top1 < confidence_threshold


def evaluate_system(ai_preds, expert_preds, true_labels, defer_mask):
    """
    Computes the combined human-AI system accuracy plus diagnostic metrics.
    defer_mask[i] == True  -> the system used the expert's prediction for i
    defer_mask[i] == False -> the system used the AI's prediction for i
    """
    true_labels = np.asarray(true_labels)
    ai_preds = np.asarray(ai_preds)
    expert_preds = np.asarray(expert_preds)
    defer_mask = np.asarray(defer_mask)

    final_preds = np.where(defer_mask, expert_preds, ai_preds)
    system_acc = float((final_preds == true_labels).mean())

    ai_only_acc = float((ai_preds == true_labels).mean())
    expert_only_acc = float((expert_preds == true_labels).mean())

    # Oracle upper bound: defer exactly when the expert is right and the AI
    # is wrong -- the best any deferral RULE could possibly achieve with
    # this AI + this expert.
    oracle_mask = (ai_preds != true_labels) & (expert_preds == true_labels)
    oracle_preds = np.where(oracle_mask, expert_preds, ai_preds)
    oracle_acc = float((oracle_preds == true_labels).mean())

    deferral_rate = float(defer_mask.mean())
    if defer_mask.sum() > 0:
        acc_on_deferred = float((final_preds[defer_mask] == true_labels[defer_mask]).mean())
    else:
        acc_on_deferred = None
    if (~defer_mask).sum() > 0:
        acc_on_kept = float((final_preds[~defer_mask] == true_labels[~defer_mask]).mean())
    else:
        acc_on_kept = None

    return {
        "system_accuracy": system_acc,
        "ai_only_accuracy": ai_only_acc,
        "expert_only_accuracy": expert_only_acc,
        "oracle_accuracy": oracle_acc,
        "deferral_rate": deferral_rate,
        "accuracy_on_deferred_subset": acc_on_deferred,
        "accuracy_on_kept_subset": acc_on_kept,
    }

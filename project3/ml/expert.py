"""
Task 2: Simulated experts for the human-AI collaboration system.

Design rationale
-----------------
A real human expert is never uniformly reliable across every topic. Someone
who reads a lot of world and sports news will label those articles almost
perfectly, but will be much shakier on business/finance jargon or technical
terminology. We model this with a *class-conditional noisy oracle*: the
expert "sees" the true label and returns it correctly with a class-specific
probability; otherwise it returns a uniformly random *wrong* label. This
gives the expert clear, structured regions of competence tied to the AG
News topics (World, Sports, Business, Sci/Tech) -- which is exactly the
kind of structure an active learning strategy (Task 4) should be able to
discover from a limited number of queries, rather than from knowing the
expert's accuracy table in advance.

A second "generalist" expert (uniform accuracy, no specialisation) is
included as a control, to make the comparison in the report concrete: it
shows why a *targeted* active learning strategy actually matters, instead
of only ever helping equally regardless of where you query.
"""

import numpy as np

from .data import LABEL_NAMES

N_CLASSES = len(LABEL_NAMES)


class BaseExpert:
    name = "base"

    def predict(self, true_labels, rng=None):
        raise NotImplementedError

    def evaluate(self, true_labels, rng=None):
        """Returns (overall_accuracy, per_class_accuracy_dict, predictions)."""
        rng = rng or np.random.default_rng(0)
        true_labels = np.asarray(true_labels)
        preds = self.predict(true_labels, rng)
        per_class = {}
        for c, cname in enumerate(LABEL_NAMES):
            mask = true_labels == c
            if mask.sum() == 0:
                continue
            per_class[cname] = float((preds[mask] == true_labels[mask]).mean())
        overall = float((preds == true_labels).mean())
        return overall, per_class, preds


class SpecialistExpert(BaseExpert):
    """Class-conditional noisy oracle: strong on World/Sports, weak on Business/Sci-Tech."""

    name = "specialist"

    def __init__(self, accuracy_by_class=None):
        # Label indices follow ml/data.py LABEL_NAMES: 0=World,1=Sports,2=Business,3=Sci/Tech
        self.accuracy_by_class_ = accuracy_by_class or {0: 0.97, 1: 0.96, 2: 0.55, 3: 0.60}

    def predict(self, true_labels, rng=None):
        rng = rng or np.random.default_rng(0)
        true_labels = np.asarray(true_labels)
        preds = np.empty_like(true_labels)
        draws = rng.random(len(true_labels))
        for i, y in enumerate(true_labels):
            p_correct = self.accuracy_by_class_.get(int(y), 0.7)
            if draws[i] < p_correct:
                preds[i] = y
            else:
                choices = [c for c in range(N_CLASSES) if c != y]
                preds[i] = rng.choice(choices)
        return preds


class GeneralistExpert(BaseExpert):
    """Uniform accuracy across all classes -- no specialisation. Used as a control."""

    name = "generalist"

    def __init__(self, accuracy=0.75):
        self.accuracy = accuracy

    def predict(self, true_labels, rng=None):
        rng = rng or np.random.default_rng(1)
        true_labels = np.asarray(true_labels)
        preds = true_labels.copy()
        wrong_mask = rng.random(len(true_labels)) >= self.accuracy
        for i in np.where(wrong_mask)[0]:
            y = true_labels[i]
            choices = [c for c in range(N_CLASSES) if c != y]
            preds[i] = rng.choice(choices)
        return preds


def get_expert(name="specialist"):
    if name == "specialist":
        return SpecialistExpert()
    if name == "generalist":
        return GeneralistExpert()
    raise ValueError(f"Unknown expert type: {name}")

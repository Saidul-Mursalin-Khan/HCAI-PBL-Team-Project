"""
Run with:
    python manage.py run_deferral

Task 3: trains the learned deferral gate using the AI classifier (Task 1)
and simulated expert labels (Task 2) on a slice of the training set, then
evaluates the combined human-AI system on the held-out test set.
"""

import json
import numpy as np
import joblib
from django.core.management.base import BaseCommand

from project3.ml.data import get_ag_news
from project3.ml.expert import SpecialistExpert
from project3.ml.deferral import DeferralGate, evaluate_system, threshold_gate_decide
from project3.ml.train_baseline import MODEL_DIR


class Command(BaseCommand):
    help = "Task 3: train and evaluate the learning-to-defer system."

    def handle(self, *args, **options):
        self.stdout.write("Loading data and trained baseline classifier...")
        train_texts, train_labels, test_texts, test_labels = get_ag_news()
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
        clf = joblib.load(MODEL_DIR / "baseline_classifier.joblib")

        X_train = vectorizer.transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        train_probs = clf.predict_proba(X_train)
        test_probs = clf.predict_proba(X_test)
        train_ai_preds = np.argmax(train_probs, axis=1)
        test_ai_preds = np.argmax(test_probs, axis=1)

        expert = SpecialistExpert()
        rng = np.random.default_rng(1)
        n = len(train_labels)
        perm = rng.permutation(n)
        # 70% of the training slice to fit the gate, 30% to calibrate the
        # deferral threshold (see ml/deferral.py for why this matters).
        fit_idx, cal_idx = perm[: int(n * 0.7)], perm[int(n * 0.7) :]

        fit_true = train_labels[fit_idx]
        cal_true = train_labels[cal_idx]
        fit_expert_preds = expert.predict(fit_true, np.random.default_rng(7))
        cal_expert_preds = expert.predict(cal_true, np.random.default_rng(8))

        gate = DeferralGate()
        gate.fit(train_probs[fit_idx], train_ai_preds[fit_idx], fit_true, fit_expert_preds)
        gate.calibrate_threshold(train_probs[cal_idx], train_ai_preds[cal_idx], cal_true, cal_expert_preds)

        test_expert_preds = expert.predict(test_labels, np.random.default_rng(42))
        defer_mask = gate.decide(test_probs)
        learned_metrics = evaluate_system(test_ai_preds, test_expert_preds, test_labels, defer_mask)
        learned_metrics["gate_threshold"] = gate.threshold_

        # Fixed-threshold baseline for comparison, at a couple of operating points.
        baseline_metrics = {}
        for t in [0.6, 0.7, 0.8]:
            mask = threshold_gate_decide(test_probs, confidence_threshold=t)
            baseline_metrics[str(t)] = evaluate_system(test_ai_preds, test_expert_preds, test_labels, mask)

        results = {"learned_gate": learned_metrics, "fixed_threshold_baseline": baseline_metrics}
        self.stdout.write(json.dumps(results, indent=2))

        out_path = MODEL_DIR / "deferral_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        self.stdout.write(self.style.SUCCESS(f"Saved learning-to-defer results to {out_path}"))

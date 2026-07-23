"""
Run with:
    python manage.py run_active_learning

Task 4: compares active learning acquisition strategies (random,
uncertainty, margin, hybrid) for choosing which training examples to send
to the expert, WITHOUT any expert labels available up front. For each
strategy and query budget, trains a deferral gate from only the queried
points and evaluates the resulting human-AI system on the test set.
"""

import json
import numpy as np
import joblib
from django.core.management.base import BaseCommand

from project3.ml.data import get_ag_news
from project3.ml.expert import SpecialistExpert
from project3.ml.active_learning import run_budget_curve
from project3.ml.train_baseline import MODEL_DIR

BUDGETS = [50, 100, 250, 500, 1000, 2000, 4000, 8000]
STRATEGIES = ["random", "uncertainty", "margin", "hybrid"]


class Command(BaseCommand):
    help = "Task 4: active learning for expert-competence discovery."

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
        test_expert_preds = expert.predict(test_labels, np.random.default_rng(42))

        results = {}
        for strategy in STRATEGIES:
            self.stdout.write(f"Running strategy: {strategy}")
            curve = run_budget_curve(
                strategy, train_probs, train_ai_preds, train_labels, expert,
                test_probs, test_ai_preds, test_labels, test_expert_preds,
                BUDGETS, seed=0,
            )
            results[strategy] = curve
            self.stdout.write(str([round(c["system_accuracy"], 4) for c in curve]))

        out_path = MODEL_DIR / "active_learning_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        self.stdout.write(self.style.SUCCESS(f"Saved active learning results to {out_path}"))

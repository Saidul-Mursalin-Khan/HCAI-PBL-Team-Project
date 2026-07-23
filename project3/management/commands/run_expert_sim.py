"""
Run with:
    python manage.py run_expert_sim

Task 2: simulates the expert(s) on the AG News test set and saves an
accuracy breakdown (overall + per class) that the Django view can display,
without needing to re-simulate on every page load.
"""

import json
import numpy as np
from django.core.management.base import BaseCommand

from project3.ml.data import get_ag_news
from project3.ml.expert import SpecialistExpert, GeneralistExpert
from project3.ml.train_baseline import MODEL_DIR


class Command(BaseCommand):
    help = "Task 2: simulate expert(s) and report their accuracy on the AG News test set."

    def handle(self, *args, **options):
        self.stdout.write("Loading AG News test set...")
        _, _, test_texts, test_labels = get_ag_news()
        test_labels = np.array(test_labels)

        rng = np.random.default_rng(42)
        experts = {"specialist": SpecialistExpert(), "generalist": GeneralistExpert()}

        results = {}
        for key, expert in experts.items():
            overall, per_class, _ = expert.evaluate(test_labels, rng)
            results[key] = {"overall_accuracy": overall, "per_class_accuracy": per_class}
            self.stdout.write(f"{key}: overall={overall:.4f} per_class={per_class}")

        out_path = MODEL_DIR / "expert_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        self.stdout.write(self.style.SUCCESS(f"Saved expert simulation results to {out_path}"))

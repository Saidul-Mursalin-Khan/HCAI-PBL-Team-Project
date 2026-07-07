"""
Run with:
    python manage.py train_baseline

This wraps ml/train_baseline.py so training is triggered like any other
Django management command, and results can later be stored/displayed
through the app (e.g. saved to a JSON file the view reads).
"""

import json
from pathlib import Path
from django.core.management.base import BaseCommand
from project3.ml.train_baseline import train_baseline, MODEL_DIR


class Command(BaseCommand):
    help = "Train the Task 1 baseline classifier on the full AG News training set."

    def handle(self, *args, **options):
        results = train_baseline()

        # Save a small results summary the Django view can read and display,
        # without needing to retrain every time the page loads.
        results_path = MODEL_DIR / "baseline_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {"accuracy": results["accuracy"], "report": results["report"]},
                f,
                indent=2,
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Baseline training complete. Accuracy: {results['accuracy']:.4f}"
            )
        )
        self.stdout.write(f"Results saved to {results_path}")

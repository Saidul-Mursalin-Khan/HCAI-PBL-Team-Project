"""
Task 1: Baseline classifier for AG News, trained on the full labeled
training set. This establishes the performance target that the
human-AI (learning-to-defer) team should match or beat later on.

Model choice: TF-IDF + Logistic Regression.
Why this is a reasonable choice for a project baseline:
- Fast to train (seconds/minutes on CPU), easy to justify and reproduce
- Strong, well-known baseline for topic classification (AG News is
  largely solvable with bag-of-words style features)
- Gives you a clean probability output (needed later for deferral
  confidence scores in Tasks 3-4)

If you want a stronger baseline later, swap this for a fine-tuned
distilbert-base-uncased -- the rest of the project (expert simulation,
deferral, active learning) doesn't depend on which classifier you pick,
only on it producing a predicted label + a confidence/probability.
"""

import time
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from .data import get_ag_news, LABEL_NAMES

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


def train_baseline():
    print("Loading AG News dataset...")
    train_texts, train_labels, test_texts, test_labels = get_ag_news()
    print(f"Train size: {len(train_texts)} | Test size: {len(test_texts)}")

    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print("Training Logistic Regression classifier...")
    start = time.time()
    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    clf.fit(X_train, train_labels)
    print(f"Training done in {time.time() - start:.1f}s")

    print("Evaluating on test set...")
    preds = clf.predict(X_test)
    acc = accuracy_score(test_labels, preds)

    print(f"\nTest accuracy: {acc:.4f}\n")
    print(classification_report(test_labels, preds, target_names=LABEL_NAMES))

    # Save both vectorizer and classifier -- you'll reuse this baseline
    # classifier as the AI side of the learning-to-defer system in Task 3.
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(clf, MODEL_DIR / "baseline_classifier.joblib")
    print(f"Saved model artifacts to {MODEL_DIR}")

    return {
        "accuracy": acc,
        "report": classification_report(
            test_labels, preds, target_names=LABEL_NAMES, output_dict=True
        ),
    }


if __name__ == "__main__":
    train_baseline()

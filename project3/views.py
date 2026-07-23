import base64
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from django.shortcuts import render

from .ml.train_baseline import MODEL_DIR
from .ml.data import LABEL_NAMES


def _load_json(name):
    path = MODEL_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def index(request):
    return render(request, "project3/index.html", {
        "title": "Project 3: Active Learning for Learning-to-Defer",
    })


def task1_baseline(request):
    results = _load_json("baseline_results.json")
    chart = None
    rows = []
    if results:
        report = results["report"]
        for name in LABEL_NAMES:
            r = report[name]
            rows.append({
                "name": name, "precision": r["precision"], "recall": r["recall"],
                "f1": r["f1-score"], "support": r["support"],
            })
        fig, ax = plt.subplots(figsize=(5, 3.5))
        f1s = [report[name]["f1-score"] for name in LABEL_NAMES]
        ax.bar(LABEL_NAMES, f1s, color="#275CB2")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1-score")
        ax.set_title("Baseline classifier: per-class F1")
        chart = _fig_to_base64(fig)
    return render(request, "project3/task1.html", {
        "title": "Task 1 -- Baseline Classifier",
        "results": results,
        "rows": rows,
        "chart": chart,
    })


def task2_expert(request):
    results = _load_json("expert_results.json")
    chart = None
    rows = []
    if results:
        for key, label in [("specialist", "Specialist"), ("generalist", "Generalist")]:
            r = results[key]
            rows.append({
                "name": label,
                "overall": r["overall_accuracy"],
                "per_class": [r["per_class_accuracy"].get(n, 0) for n in LABEL_NAMES],
            })
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        width = 0.35
        x = np.arange(len(LABEL_NAMES))
        spec_vals = [results["specialist"]["per_class_accuracy"].get(n, 0) for n in LABEL_NAMES]
        gen_vals = [results["generalist"]["per_class_accuracy"].get(n, 0) for n in LABEL_NAMES]
        ax.bar(x - width / 2, spec_vals, width, label="Specialist expert", color="#275CB2")
        ax.bar(x + width / 2, gen_vals, width, label="Generalist expert", color="#9BB6DE")
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_NAMES)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Simulated expert accuracy by topic")
        ax.legend()
        chart = _fig_to_base64(fig)
    return render(request, "project3/task2.html", {
        "title": "Task 2 -- Simulated Expert",
        "results": results,
        "rows": rows,
        "label_names": LABEL_NAMES,
        "chart": chart,
    })


def task3_deferral(request):
    results = _load_json("deferral_results.json")
    chart = None
    if results:
        gate = results["learned_gate"]
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        labels = ["AI only", "Expert only", "Learned\ndeferral", "Oracle\n(upper bound)"]
        values = [
            gate["ai_only_accuracy"], gate["expert_only_accuracy"],
            gate["system_accuracy"], gate["oracle_accuracy"],
        ]
        colors = ["#9BB6DE", "#9BB6DE", "#275CB2", "#1B3E7A"]
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test accuracy")
        ax.set_title("Learning-to-defer: system vs. baselines")
        chart = _fig_to_base64(fig)
    return render(request, "project3/task3.html", {
        "title": "Task 3 -- Learning to Defer",
        "results": results,
        "chart": chart,
    })


def task4_active_learning(request):
    results = _load_json("active_learning_results.json")
    chart = None
    table_rows = []
    strategy_names = []
    if results:
        strategy_names = list(results.keys())
        budgets = [c["n_queries"] for c in results[strategy_names[0]]]
        for i, b in enumerate(budgets):
            table_rows.append({
                "budget": b,
                "accs": [round(results[s][i]["system_accuracy"], 4) for s in strategy_names],
            })
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {"random": "#9BB6DE", "uncertainty": "#F2A65A", "margin": "#E0574C", "hybrid": "#275CB2"}
        for strategy, curve in results.items():
            xs = [c["n_queries"] for c in curve]
            ys = [c["system_accuracy"] for c in curve]
            ax.plot(xs, ys, marker="o", label=strategy, color=colors.get(strategy))
        ax.set_xscale("log")
        ax.set_xlabel("Number of expert queries (log scale)")
        ax.set_ylabel("System test accuracy")
        ax.set_title("Active learning: accuracy vs. query budget")
        ax.legend()
        chart = _fig_to_base64(fig)
    return render(request, "project3/task4.html", {
        "title": "Task 4 -- Active Learning for Expert-Competence Discovery",
        "results": results,
        "table_rows": table_rows,
        "strategy_names": strategy_names,
        "chart": chart,
    })


def task5_try_it(request):
    """Optional Task 5: a minimal interactive demo. The visitor pastes a
    news snippet, the trained baseline classifies it, and the learned
    deferral gate's threshold decides whether the system would defer to a
    human expert. This is a lightweight stand-in for the full
    expert-in-the-loop interface the assignment describes as an optional
    extension."""
    context = {"title": "Task 5 (Optional) -- Try It Yourself"}
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        if text:
            vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
            clf = joblib.load(MODEL_DIR / "baseline_classifier.joblib")
            X = vectorizer.transform([text])
            probs = clf.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))

            deferral_results = _load_json("deferral_results.json")
            threshold = deferral_results["learned_gate"]["gate_threshold"] if deferral_results else 0.5
            top1 = float(np.max(probs))

            context.update({
                "submitted_text": text,
                "predicted_label": LABEL_NAMES[pred_idx],
                "confidence": round(top1, 3),
                "probs": {LABEL_NAMES[i]: round(float(p), 3) for i, p in enumerate(probs)},
                "would_defer": bool(top1 < threshold),
                "threshold": round(float(threshold), 3),
            })
    return render(request, "project3/task5.html", context)

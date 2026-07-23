import numpy as np
import joblib
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse

from .ml.train_baseline import MODEL_DIR
from .ml.data import LABEL_NAMES
from . import charts
from .report import build_report_pdf

# The guided roadmap. Each entry: (step number, full label, short label).
STEP_DEFS = [
    (1, "Baseline Classifier", "Baseline"),
    (2, "Simulated Expert", "Expert"),
    (3, "Learning to Defer", "Defer"),
    (4, "Active Learning", "Active"),
    (5, "Try It Yourself", "Try It"),
]
LAST_STEP = STEP_DEFS[-1][0]
SESSION_KEY = "p3_unlocked"


# --- Roadmap / progress helpers -----------------------------------------

def _unlocked(request):
    """Highest step the visitor has unlocked (1-based). Defaults to 1."""
    try:
        return max(1, int(request.session.get(SESSION_KEY, 1)))
    except (TypeError, ValueError):
        return 1


def build_steps(request, current):
    """Per-step metadata for the roadmap partial.

    `current` is the step number of the page being viewed (0 for the
    overview). `state` is one of: locked / current / done / available.
    """
    unlocked = _unlocked(request)
    steps = []
    for number, label, short in STEP_DEFS:
        if number > unlocked:
            state = "locked"
        elif number == current:
            state = "current"
        elif number < unlocked:
            state = "done"
        else:
            state = "available"
        steps.append({
            "number": number,
            "label": label,
            "short": short,
            "url": reverse(f"project3:task{number}") if state != "locked" else "",
            "state": state,
        })
    return steps


def _pct(value, digits=0):
    """Turn a 0..1 fraction into a percentage number for display."""
    n = round(value * 100, digits)
    return int(n) if digits == 0 else n


def _guard(request, step):
    """If `step` is not yet unlocked, return a redirect to the current
    (highest unlocked) step. Otherwise return None."""
    unlocked = _unlocked(request)
    if step > unlocked:
        target = min(unlocked, LAST_STEP)
        return redirect(reverse(f"project3:task{target}"))
    return None


def _task_context(request, step, title, blurb):
    """Common context shared by every task page."""
    unlocked = _unlocked(request)
    prev_url = reverse(f"project3:task{step - 1}") if step > 1 else reverse("project3:index")
    next_url = reverse(f"project3:task{step + 1}") if step < LAST_STEP else None
    return {
        "title": title,
        "blurb": blurb,
        "step": step,
        "last_step": LAST_STEP,
        "steps": build_steps(request, step),
        "is_done": step < unlocked,           # already completed before
        "advance_url": reverse("project3:advance", args=[step]),
        "prev_url": prev_url,
        "next_url": next_url,
    }


def advance(request, step):
    """Mark `step` complete, unlock the next one, and move the visitor on."""
    unlocked = _unlocked(request)
    request.session[SESSION_KEY] = max(unlocked, step + 1)
    if step < LAST_STEP:
        return redirect(reverse(f"project3:task{step + 1}"))
    return redirect(reverse("project3:index") + "#roadmap")


def reset_progress(request):
    """Clear roadmap progress and return to the overview."""
    request.session.pop(SESSION_KEY, None)
    return redirect(reverse("project3:index"))


# --- Pages ---------------------------------------------------------------

def index(request):
    unlocked = _unlocked(request)
    current = min(unlocked, LAST_STEP)
    completed = max(0, unlocked - 1)
    return render(request, "project3/index.html", {
        "title": "Active Learning for Learning-to-Defer",
        "steps": build_steps(request, 0),
        "current_step": current,
        "continue_url": reverse(f"project3:task{current}"),
        "completed": completed,
        "total_steps": LAST_STEP,
        "started": unlocked > 1,
        "all_done": unlocked > LAST_STEP,
    })


def task1_baseline(request):
    redirect_response = _guard(request, 1)
    if redirect_response:
        return redirect_response
    results = charts.load_result("baseline_results.json")
    rows, chart = [], None
    if results:
        report = results["report"]
        for name in LABEL_NAMES:
            r = report[name]
            rows.append({
                "name": name, "precision": r["precision"], "recall": r["recall"],
                "f1": r["f1-score"], "support": r["support"],
            })
        chart = charts.fig_to_base64(charts.baseline_fig(results))
    ctx = _task_context(
        request, 1, "Baseline Classifier",
        "Teach the computer to sort news stories on its own — this is the "
        "score the human + AI team will try to beat later.")
    ctx.update({
        "results": results, "rows": rows, "chart": chart,
        "accuracy_pct": _pct(results["accuracy"]) if results else None,
    })
    return render(request, "project3/task1.html", ctx)


def task2_expert(request):
    redirect_response = _guard(request, 2)
    if redirect_response:
        return redirect_response
    results = charts.load_result("expert_results.json")
    rows, chart = [], None
    if results:
        for key, label in [("specialist", "Specialist"), ("generalist", "Generalist")]:
            r = results[key]
            rows.append({
                "name": label,
                "overall": r["overall_accuracy"],
                "overall_pct": _pct(r["overall_accuracy"]),
                "per_class": [r["per_class_accuracy"].get(n, 0) for n in LABEL_NAMES],
            })
        chart = charts.fig_to_base64(charts.expert_fig(results))
    ctx = _task_context(
        request, 2, "The Human Expert",
        "Invent a pretend human helper who is brilliant at some topics and "
        "shaky at others — just like a real person.")
    ctx.update({"results": results, "rows": rows, "label_names": LABEL_NAMES, "chart": chart})
    return render(request, "project3/task2.html", ctx)


def task3_deferral(request):
    redirect_response = _guard(request, 3)
    if redirect_response:
        return redirect_response
    results = charts.load_result("deferral_results.json")
    chart, gate_pct = None, {}
    if results:
        chart = charts.fig_to_base64(charts.deferral_fig(results))
        g = results["learned_gate"]
        gate_pct = {
            "ai_only": _pct(g["ai_only_accuracy"], 1),
            "expert_only": _pct(g["expert_only_accuracy"], 1),
            "system": _pct(g["system_accuracy"], 1),
            "oracle": _pct(g["oracle_accuracy"], 1),
            "deferral": _pct(g["deferral_rate"]),
        }
    ctx = _task_context(
        request, 3, "Knowing When to Ask for Help",
        "Let the computer learn when to answer by itself and when to hand a "
        "tricky story to the human expert.")
    ctx.update({"results": results, "chart": chart, "gate_pct": gate_pct})
    return render(request, "project3/task3.html", ctx)


def task4_active_learning(request):
    redirect_response = _guard(request, 4)
    if redirect_response:
        return redirect_response
    results = charts.load_result("active_learning_results.json")
    chart, table_rows, strategy_names = None, [], []
    if results:
        strategy_names = list(results.keys())
        budgets = [c["n_queries"] for c in results[strategy_names[0]]]
        for i, b in enumerate(budgets):
            table_rows.append({
                "budget": b,
                "accs": [round(results[s][i]["system_accuracy"], 4) for s in strategy_names],
            })
        chart = charts.fig_to_base64(charts.active_learning_fig(results))
    ctx = _task_context(
        request, 4, "Asking Smart Questions",
        "The expert's time is precious. Learn which few stories are worth "
        "asking about so we get the most help from the fewest questions.")
    ctx.update({
        "results": results, "table_rows": table_rows,
        "strategy_names": strategy_names, "chart": chart,
    })
    return render(request, "project3/task4.html", ctx)


def task5_try_it(request):
    redirect_response = _guard(request, 5)
    if redirect_response:
        return redirect_response
    ctx = _task_context(
        request, 5, "Try It Yourself",
        "Your turn! Paste any news snippet and watch the team decide whether "
        "the AI answers or asks the human expert.")
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        if text:
            vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
            clf = joblib.load(MODEL_DIR / "baseline_classifier.joblib")
            X = vectorizer.transform([text])
            probs = clf.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))

            deferral_results = charts.load_result("deferral_results.json")
            threshold = (
                deferral_results["learned_gate"]["gate_threshold"]
                if deferral_results else 0.5
            )
            top1 = float(np.max(probs))
            ctx.update({
                "submitted_text": text,
                "predicted_label": LABEL_NAMES[pred_idx],
                "confidence": round(top1, 3),
                "confidence_pct": round(top1 * 100),
                "probs": [
                    {"name": LABEL_NAMES[i], "p": round(float(p), 3),
                     "pct": round(float(p) * 100)}
                    for i, p in enumerate(probs)
                ],
                "would_defer": bool(top1 < threshold),
                "threshold": round(float(threshold), 3),
            })
    return render(request, "project3/task5.html", ctx)


def report(request):
    """Generate and download the PDF report, live from the results JSON."""
    pdf_bytes = build_report_pdf()
    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="Project3_Report.pdf"'
    return response

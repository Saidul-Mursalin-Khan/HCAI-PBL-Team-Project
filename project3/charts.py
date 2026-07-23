"""Shared chart + data helpers for Project 3.

Both the web views (which embed charts as base64 PNGs) and the PDF report
generator (which writes the same figures into a multi-page PDF) build their
figures here, so the on-screen visuals and the downloadable report stay in
sync and read from the same results JSON.
"""

import base64
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .ml.train_baseline import MODEL_DIR
from .ml.data import LABEL_NAMES

# Project palette (kept in sync with the site's --blue tokens).
BLUE = "#275CB2"
BLUE_DARK = "#1B3E7A"
BLUE_LIGHT = "#9BB6DE"
ORANGE = "#F2A65A"
RED = "#E0574C"
GREEN = "#1D9E75"

STRATEGY_COLORS = {
    "random": BLUE_LIGHT,
    "uncertainty": ORANGE,
    "margin": RED,
    "hybrid": BLUE,
}


def load_result(name):
    """Load a saved results JSON file, or return None if it is missing."""
    path = MODEL_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fig_to_base64(fig):
    """Render a matplotlib figure to a base64 PNG string for inline <img>."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def fig_to_png_bytes(fig, dpi=150):
    """Render a matplotlib figure to raw PNG bytes (used by the PDF report)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# --- Figure builders -----------------------------------------------------
# Each returns a matplotlib Figure. Views base64-encode it; the report
# writes it straight into a PdfPages document.


def baseline_fig(results):
    report = results["report"]
    f1s = [report[name]["f1-score"] for name in LABEL_NAMES]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(LABEL_NAMES, f1s, color=BLUE)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title("Baseline classifier: per-topic F1")
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    return fig


def expert_fig(results):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    width = 0.35
    x = np.arange(len(LABEL_NAMES))
    spec = [results["specialist"]["per_class_accuracy"].get(n, 0) for n in LABEL_NAMES]
    gen = [results["generalist"]["per_class_accuracy"].get(n, 0) for n in LABEL_NAMES]
    ax.bar(x - width / 2, spec, width, label="Specialist expert", color=BLUE)
    ax.bar(x + width / 2, gen, width, label="Generalist expert", color=BLUE_LIGHT)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Simulated expert accuracy by topic")
    ax.legend()
    return fig


def deferral_fig(results):
    gate = results["learned_gate"]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    labels = ["AI only", "Expert only", "Learned\ndeferral", "Oracle\n(upper bound)"]
    values = [
        gate["ai_only_accuracy"], gate["expert_only_accuracy"],
        gate["system_accuracy"], gate["oracle_accuracy"],
    ]
    colors = [BLUE_LIGHT, BLUE_LIGHT, BLUE, BLUE_DARK]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Learning-to-defer: system vs. baselines")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    return fig


def active_learning_fig(results):
    fig, ax = plt.subplots(figsize=(6, 4))
    for strategy, curve in results.items():
        xs = [c["n_queries"] for c in curve]
        ys = [c["system_accuracy"] for c in curve]
        ax.plot(xs, ys, marker="o", label=strategy, color=STRATEGY_COLORS.get(strategy))
    ax.set_xscale("log")
    ax.set_xlabel("Number of expert queries (log scale)")
    ax.set_ylabel("System test accuracy")
    ax.set_title("Active learning: accuracy vs. query budget")
    ax.legend()
    return fig

"""Generate the Project 3 PDF report, live from the saved results JSON.

Uses matplotlib's PdfPages backend so no extra dependency is needed. The
charts are the exact same figures shown on the website (built in charts.py),
so the report always matches what the visitor sees on screen.
"""

import io
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from django.utils import timezone

from . import charts
from .ml.data import LABEL_NAMES

A4 = (8.27, 11.69)  # inches, portrait
INK = "#172033"
MUTED = "#4e5d73"
LINE = "#cdd8e7"
BLUE = charts.BLUE
BLUE_DARK = charts.BLUE_DARK


# --- low-level page primitives ------------------------------------------

def _page(pdf):
    """Start a new A4 page. Returns (fig, ax) where ax spans the whole page
    in 0..1 coordinates with the axis hidden."""
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _header(ax, kicker, title):
    ax.add_patch(Rectangle((0, 0.918), 1, 0.082, color=BLUE_DARK, zorder=0))
    ax.add_patch(Rectangle((0, 0.912), 1, 0.006, color=BLUE, zorder=0))
    ax.text(0.07, 0.972, kicker.upper(), color="#bcd0f0", fontsize=10,
            fontweight="bold", va="center")
    ax.text(0.07, 0.945, title, color="white", fontsize=19,
            fontweight="bold", va="center")


def _footer(ax, page_no):
    ax.plot([0.07, 0.93], [0.05, 0.05], color=LINE, lw=0.8)
    ax.text(0.07, 0.032, "Project 3 — Active Learning for Learning-to-Defer",
            color=MUTED, fontsize=8, va="center")
    ax.text(0.93, 0.032, f"Page {page_no}", color=MUTED, fontsize=8,
            va="center", ha="right")


def _para(ax, y, text, size=10.5, color=INK, width=96, weight="normal",
          lh=0.019):
    """Draw a wrapped paragraph starting at top y; return the new y cursor."""
    lines = []
    for block in text.split("\n"):
        wrapped = textwrap.wrap(block, width=width) or [""]
        lines.extend(wrapped)
    for line in lines:
        ax.text(0.07, y, line, fontsize=size, color=color, va="top",
                fontweight=weight)
        y -= lh
    return y - lh * 0.4


def _heading(ax, y, text):
    ax.text(0.07, y, text, fontsize=13, color=BLUE_DARK, fontweight="bold",
            va="top")
    return y - 0.028


def _chart(fig, fig_obj, box):
    """Place a chart figure (from charts.py) into `box` = [l, b, w, h]."""
    png = charts.fig_to_png_bytes(fig_obj)
    img = mpimg.imread(io.BytesIO(png))
    ax_img = fig.add_axes(box)
    ax_img.imshow(img)
    ax_img.axis("off")


def _table(fig, box, col_labels, rows, col_widths=None):
    ax_t = fig.add_axes(box)
    ax_t.axis("off")
    tbl = ax_t.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(LINE)
        if r == 0:
            cell.set_facecolor(BLUE_DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f7fb")
    return ax_t


# --- content pages ------------------------------------------------------

def _cover(pdf):
    fig, ax = _page(pdf)
    ax.add_patch(Rectangle((0, 0.62), 1, 0.38, color=BLUE_DARK, zorder=0))
    ax.add_patch(Rectangle((0, 0.608), 1, 0.012, color=BLUE, zorder=0))
    ax.text(0.5, 0.86, "PROJECT 3", color="#bcd0f0", fontsize=15,
            fontweight="bold", ha="center")
    ax.text(0.5, 0.79, "Active Learning for", color="white", fontsize=30,
            fontweight="bold", ha="center")
    ax.text(0.5, 0.735, "Learning-to-Defer", color="white", fontsize=30,
            fontweight="bold", ha="center")
    ax.text(0.5, 0.66, "Human-Centric Artificial Intelligence", color="#dce7f7",
            fontsize=13, ha="center", style="italic")

    y = 0.54
    y = _para(ax, y,
              "This report documents a human–AI team that classifies news "
              "stories. The AI answers the easy cases itself and learns to "
              "defer the hard ones to a human expert — and it learns whom to "
              "ask, and when, using as few expert questions as possible.",
              size=11.5, width=88)

    # Headline results pulled live from the saved metrics.
    baseline = charts.load_result("baseline_results.json")
    deferral = charts.load_result("deferral_results.json")
    facts = []
    if baseline:
        facts.append(("AI accuracy (baseline)",
                      f"{baseline['accuracy'] * 100:.1f}%"))
    if deferral:
        g = deferral["learned_gate"]
        facts.append(("Human + AI team accuracy",
                      f"{g['system_accuracy'] * 100:.1f}%"))
        facts.append(("Questions sent to the expert",
                      f"{g['deferral_rate'] * 100:.1f}% of stories"))
    facts.append(("Dataset", "AG News — 120,000 train / 7,600 test, 4 topics"))

    box_y = 0.36
    ax.add_patch(Rectangle((0.07, box_y - 0.02 - 0.05 * len(facts)), 0.86,
                           0.04 + 0.05 * len(facts), fill=False,
                           edgecolor=LINE, lw=1))
    yy = box_y
    for label, value in facts:
        ax.text(0.11, yy, label, fontsize=10.5, color=MUTED, va="center")
        ax.text(0.89, yy, value, fontsize=11.5, color=BLUE_DARK, va="center",
                ha="right", fontweight="bold")
        yy -= 0.05

    ax.text(0.07, 0.08, "Generated " + timezone.localtime().strftime("%d %B %Y"),
            fontsize=9, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def _task1_page(pdf, page_no):
    fig, ax = _page(pdf)
    _header(ax, "Task 1", "Baseline Classifier")
    results = charts.load_result("baseline_results.json")
    y = 0.87
    y = _heading(ax, y, "What we did")
    y = _para(ax, y,
              "We trained a text classifier on the full AG News training set "
              "(120,000 articles) and measured it on 7,600 unseen articles. "
              "This score is the target the human–AI team must match or beat.")
    y = _heading(ax, y - 0.005, "Design choice")
    y = _para(ax, y,
              "We chose TF-IDF features with a Logistic Regression classifier. "
              "It is fast, strong on topic classification, and — crucially — "
              "gives calibrated probabilities. Those probabilities become the "
              "confidence signal the later deferral gate relies on.")
    if results:
        acc = results["accuracy"]
        ax.text(0.07, y - 0.01, f"Test accuracy: {acc * 100:.2f}%",
                fontsize=13, color=BLUE_DARK, fontweight="bold", va="top")
        rep = results["report"]
        rows = [[n, f"{rep[n]['precision']:.3f}", f"{rep[n]['recall']:.3f}",
                 f"{rep[n]['f1-score']:.3f}", f"{int(rep[n]['support'])}"]
                for n in LABEL_NAMES]
        _table(fig, [0.07, 0.40, 0.5, 0.18],
               ["Topic", "Precision", "Recall", "F1", "Support"], rows)
        _chart(fig, charts.baseline_fig(results), [0.58, 0.36, 0.37, 0.24])
    else:
        _para(ax, y, "No results found. Run: python manage.py train_baseline")
    _footer(ax, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def _task2_page(pdf, page_no):
    fig, ax = _page(pdf)
    _header(ax, "Task 2", "Simulating a Human Expert")
    results = charts.load_result("expert_results.json")
    y = 0.87
    y = _heading(ax, y, "What we did")
    y = _para(ax, y,
              "A real expert is not perfect and not evenly skilled. We built "
              "two simulated experts as class-conditional noisy oracles: a "
              "Specialist who is strong on World and Sports news but weak on "
              "Business and Sci/Tech, and a Generalist with uniform accuracy "
              "as a control.")
    y = _heading(ax, y - 0.005, "Why this matters")
    y = _para(ax, y,
              "The Specialist's uneven skill is deliberate. It gives the "
              "deferral system (Task 3) and the active-learning strategy "
              "(Task 4) a non-trivial competence profile to discover, instead "
              "of an expert who is simply better everywhere.")
    if results:
        rows = []
        for key, label in [("specialist", "Specialist"), ("generalist", "Generalist")]:
            r = results[key]
            rows.append([label, f"{r['overall_accuracy']:.3f}"] +
                        [f"{r['per_class_accuracy'].get(n, 0):.3f}" for n in LABEL_NAMES])
        _table(fig, [0.07, 0.44, 0.86, 0.12],
               ["Expert", "Overall"] + LABEL_NAMES, rows)
        _chart(fig, charts.expert_fig(results), [0.22, 0.10, 0.56, 0.30])
    else:
        _para(ax, y, "No results found. Run: python manage.py run_expert_sim")
    _footer(ax, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def _task3_page(pdf, page_no):
    fig, ax = _page(pdf)
    _header(ax, "Task 3", "Learning to Defer")
    results = charts.load_result("deferral_results.json")
    y = 0.87
    y = _heading(ax, y, "What we did")
    y = _para(ax, y,
              "With both AI predictions and expert labels available, we "
              "trained a small logistic-regression 'gate' that decides, per "
              "story, whether to trust the AI or defer to the expert. The gate "
              "is trained to defer exactly when doing so flips a wrong AI "
              "answer into a right one, and its threshold is calibrated on a "
              "held-out split to maximise combined accuracy.")
    y = _heading(ax, y - 0.005, "How we evaluate it")
    y = _para(ax, y,
              "We report not just accuracy but the quality of the deferral "
              "decisions: how often the system defers, and how accurate it is "
              "on the kept vs. deferred stories.")
    if results:
        g = results["learned_gate"]
        rows = [
            ["AI only", f"{g['ai_only_accuracy']:.4f}"],
            ["Expert only", f"{g['expert_only_accuracy']:.4f}"],
            ["Learned deferral system", f"{g['system_accuracy']:.4f}"],
            ["Oracle (upper bound)", f"{g['oracle_accuracy']:.4f}"],
            ["Deferral rate", f"{g['deferral_rate']:.3f}"],
            ["Accuracy on kept stories", f"{g['accuracy_on_kept_subset']:.3f}"],
            ["Accuracy on deferred stories", f"{g['accuracy_on_deferred_subset']:.3f}"],
            ["Calibrated gate threshold", f"{g['gate_threshold']:.3f}"],
        ]
        _para(ax, y - 0.005,
              "The team beats AI-only accuracy by deferring sparingly — only "
              "the AI's least-confident stories. Accuracy on the deferred "
              "subset is low because those are the hardest cases for everyone, "
              "but the expert still edges out the AI there.", width=96)
        _table(fig, [0.07, 0.30, 0.5, 0.16], ["Metric", "Value"], rows,
               col_widths=[0.78, 0.22])
        _chart(fig, charts.deferral_fig(results), [0.6, 0.27, 0.36, 0.22])
    else:
        _para(ax, y, "No results found. Run: python manage.py run_deferral")
    _footer(ax, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def _task4_page(pdf, page_no):
    fig, ax = _page(pdf)
    _header(ax, "Task 4", "Active Learning for Expert Discovery")
    results = charts.load_result("active_learning_results.json")
    y = 0.87
    y = _heading(ax, y, "What we did")
    y = _para(ax, y,
              "Now we start with zero expert labels and may query the expert "
              "only a limited number of times. We compare four strategies for "
              "choosing which stories to ask about: random (control), "
              "uncertainty sampling (the AI's least-confident stories), margin "
              "sampling (smallest gap between the top two guesses), and a "
              "hybrid of random + uncertainty.")
    y = _heading(ax, y - 0.005, "Key finding")
    y = _para(ax, y,
              "At tiny budgets, pure uncertainty/margin sampling can collapse "
              "into an 'always defer' policy: it only ever sees the AI's "
              "hardest stories, so the gate never learns when deferring does "
              "NOT help. The hybrid strategy keeps some random exploration and "
              "avoids this failure, at a small efficiency cost once budgets grow.")
    if results:
        strategies = list(results.keys())
        budgets = [c["n_queries"] for c in results[strategies[0]]]
        rows = [[str(b)] + [f"{results[s][i]['system_accuracy']:.4f}"
                            for s in strategies]
                for i, b in enumerate(budgets)]
        _table(fig, [0.07, 0.30, 0.86, 0.16], ["Queries"] + strategies, rows)
        _chart(fig, charts.active_learning_fig(results), [0.22, 0.04, 0.56, 0.24])
    else:
        _para(ax, y, "No results found. Run: python manage.py run_active_learning")
    _footer(ax, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def _conclusion_page(pdf, page_no):
    fig, ax = _page(pdf)
    _header(ax, "Conclusion", "What the Team Achieved")
    baseline = charts.load_result("baseline_results.json")
    deferral = charts.load_result("deferral_results.json")
    y = 0.87
    if baseline and deferral:
        ai = baseline["accuracy"] * 100
        sys = deferral["learned_gate"]["system_accuracy"] * 100
        rate = deferral["learned_gate"]["deferral_rate"] * 100
        y = _para(ax, y,
                  f"The AI on its own reaches {ai:.1f}% accuracy. By learning "
                  f"to hand just {rate:.1f}% of the hardest stories to a human "
                  f"expert, the human–AI team reaches {sys:.1f}% — matching or "
                  "beating the AI while asking for help only when it truly "
                  "pays off.", size=11.5, width=86)
    y = _para(ax, y - 0.005,
              "The project shows the full human-in-the-loop pipeline: train a "
              "strong baseline, model a realistically imperfect expert, learn "
              "a calibrated deferral policy, and — when expert labels are "
              "scarce — use active learning to discover the expert's "
              "competence with as few questions as possible.", size=11.5, width=86)
    y = _heading(ax, y - 0.01, "Reproducing these numbers")
    y = _para(ax, y,
              "python manage.py train_baseline\n"
              "python manage.py run_expert_sim\n"
              "python manage.py run_deferral\n"
              "python manage.py run_active_learning",
              size=10, color=MUTED, width=90)
    _footer(ax, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def build_report_pdf():
    """Build the full report and return it as PDF bytes."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _cover(pdf)
        _task1_page(pdf, 2)
        _task2_page(pdf, 3)
        _task3_page(pdf, 4)
        _task4_page(pdf, 5)
        _conclusion_page(pdf, 6)
    buf.seek(0)
    return buf.getvalue()

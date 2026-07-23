# Integrating this project3 into the team repo

This folder is a complete, drop-in replacement for `project3/` in the team
repo. Task 1 (baseline classifier) was already done by a teammate and is
preserved as-is; Tasks 2, 3, 4, and the optional Task 5 have been added.

## Steps

1. Replace the existing `project3/` folder in the repo with this one.
2. Add one line to `requirements.txt`:
   ```
   datasets
   ```
   (needed by `project3/ml/data.py` to load AG News from Hugging Face --
   it was missing even for the existing Task 1 code).
3. Two stale comments/labels from an earlier topic assignment should be
   fixed (harmless either way, but misleading):
   - `pbl/settings.py`: the `'project3',   # Explainability` comment
   - `pbl/urls.py`: the `# Explainability` comment on the project3 include
   - `home/views.py`: the nav card currently reads
     `"Project 3: Explainability"` -- update to
     `"Project 3: Active Learning for Learning-to-Defer"`
4. Run the four management commands in order (each is idempotent and
   saves its results as JSON under `project3/ml/saved_models/`, which the
   views read from):
   ```
   python manage.py train_baseline       # Task 1 (teammate's, already done)
   python manage.py run_expert_sim       # Task 2
   python manage.py run_deferral         # Task 3
   python manage.py run_active_learning  # Task 4 (can take a few minutes)
   ```
5. `python manage.py runserver` and visit `/project3/` -- it now has five
   pages (one per task) plus a "Download PDF report" button in the nav
   bar on every page, per the assignment's requirement that the report be
   reachable from the project interface.

## What's included

- `ml/expert.py` -- Task 2: two simulated experts (a class-conditional
  "Specialist" and a uniform "Generalist" control).
- `ml/deferral.py` -- Task 3: a learned, threshold-calibrated deferral
  gate, plus a fixed-threshold baseline for comparison.
- `ml/active_learning.py` -- Task 4: four query-acquisition strategies
  (random, uncertainty, margin, hybrid) evaluated across a range of query
  budgets.
- `management/commands/run_expert_sim.py`, `run_deferral.py`,
  `run_active_learning.py` -- one command per task, following the same
  pattern as the existing `train_baseline.py`.
- `views.py` / `urls.py` / `templates/project3/*.html` -- one page per
  task (`/project3/task1/` .. `/project3/task5/`), each with a results
  table and a matplotlib chart, in the site's existing visual style.
- `static/project3/style.css` -- small stylesheet for the nav bar, tables,
  and charts.
- `static/project3/Project3_Report.pdf` -- the required PDF report,
  linked from the "Download PDF report" button.

## A finding worth highlighting in a demo/defense

At very small expert-query budgets, pure uncertainty/margin-based active
learning can collapse into an "always defer" policy (see Task 4 page and
report section 4) -- a genuine, explainable failure mode, not a bug. The
hybrid strategy was added specifically to fix it. This is good material
if you're asked to explain a design decision.

## Note on the numbers currently baked into the JSON files

Tasks 2-4's saved JSON results were generated using a 30,000-row random
subset of the 120,000-row training set (for fast iteration while building
this). Task 1's own numbers are already from the full training set. Re-run
the three new management commands after `train_baseline` has been run on
the full set, to make all numbers consistent before final submission --
the qualitative conclusions (deferral beats AI-only, hybrid avoids the
degenerate-gate failure mode, etc.) held up the same way in early testing
on the full set and should not change.

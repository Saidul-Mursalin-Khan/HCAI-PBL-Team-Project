# Project 2 - Explainability

*Human-Centric Artificial Intelligence – Team Project*

This Django app implements the HCAI Project 2 explainability assignment within the team project.

---

## Table of Contents

- [What is included](#what-is-included)
- [Setup](#setup)
  - [Windows](#windows)
  - [Linux / macOS](#linux--macos)
- [Prepare and verify](#prepare-and-verify)
- [Run the server](#run-the-server)
- [Additional Project 2 commands](#additional-project-2-commands)
- [Model definitions](#model-definitions)
- [Data provenance](#data-provenance)
- [Report](#report)

---

## What is included

- **Task 1** – Decision tree on the Palmer Penguins dataset, with a user interface showing the tree, its test accuracy, and the number of leaves.
- **Task 2** – Several models with varying degrees of regularization, and a slider (λ) that selects the model maximizing `acc_test − λ · Ω(f)`, where Ω(f) is the number of leaves.
- **Task 3** – Logistic regression model class (used alongside the tree for comparison).
- **Task 4** – Counterfactual explanations: select an example and a target label, and view the best k counterfactuals ranked by MAD-weighted L¹-distance. Handles decimal, binary, and categorical features with appropriate noise.
- **Task 5** – Global model-agnostic feature effect plots: select one of the four numerical features and view both PDP and ALE plots for each species (three curves per plot). PDP and ALE are implemented from scratch; exact partial derivatives for logistic regression, discretization for the tree.
- Linked interface: model class (tree or logistic regression) and λ can be freely selected, and all regions (counterfactuals, feature effect plots) update accordingly.
- CSRF-protected server-side validation.
- Unit, integration, CSRF, and end-to-end tests.
- Downloadable methods and results report, linked from every page.

---

## Setup

### Windows

From the repository root, create a virtual environment and install the shared requirements:

```powershell
py -m venv env
.\env\Scripts\python.exe -m pip install --upgrade pip
.\env\Scripts\python.exe -m pip install -r requirements.txt
```
Select `env\Scripts\python.exe` through Python: Select Interpreter in VS Code.

### Linux / macOS
From the repository root, create a virtual environment and install the shared requirements:
```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
```markdown
If you use VS Code, select `env/bin/pythonthrough Python: Select Interpreter.
```
---
## Prepare and verify

### Windows
```powershell
.\env\Scripts\python.exe manage.py migrate
.\env\Scripts\python.exe manage.py check
.\env\Scripts\python.exe manage.py makemigrations --check --dry-run
.\env\Scripts\python.exe manage.py test project2
```
### Linux / macOS
```bash
python manage.py migrate
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py test project2
```
---
## Run the server
## Windows
```powershell
.\env\Scripts\python.exe manage.py runserver
```
###  Linux / macOS
```bash
python manage.py runserver
```

Open http://127.0.0.1:8000/project2/.

---

## Model definitions
### Decision tree and regularization
The decision tree is fit with DecisionTreeClassifier from scikit-learn. Regularization is controlled by max_leaf_nodes. For a given λ, the selected model maximizes:

```text
acc_test − λ · Ω(f)
where Ω(f) is the number of leaves and acc_test is the test accuracy.
```

### Counterfactual generation
Given an example x and a target class:

1)Randomly sample N points locally around x.

2)Keep those whose prediction is the target class.

3)Rank them by MAD-weighted L¹-distance to x.

4)Display the best k counterfactuals.

For decimal features, add Gaussian noise. For binary features, flip with small probability. For categorical features, sample from the other categories.

If no counterfactuals are found, increase N and/or adjust the sampling variance iteratively.

### PDP and ALE
- PDP: average model prediction over the marginal distribution of the selected feature.

- ALE: accumulate local differences in predictions over a grid of feature values.

Exact partial derivatives are available for logistic regression. For the decision tree, discretization is used to approximate derivatives.
```markdown
Both PDP and ALE are implemented from scratch (no external libraries for these computations).
```
---

## Data provenance
Palmer Penguins dataset, obtained via the palmerpenguins Python package.

Features: species, sex, bill_length_mm, flipper_length_mm, island, year, bill_depth_mm, body_mass_g.
```markdown
Target: species with three classes: Adelie, Gentoo, Chinstrap.
```
---
## Report
A downloadable methods and results report is available at:

```text
/project2/report/
```
The report is linked from every page of the Project 2 interface.




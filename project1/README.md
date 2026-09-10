# Project 1 - Automated Machine Learning

*Human-Centric Artificial Intelligence – Team Project*

This Django app implements the HCAI Project 1 supervised learning interface within the team project.

---

## Table of Contents

- [What is included](#what-is-included)
- [Setup](#setup)
  - [Windows](#windows)
  - [Linux / macOS](#linux--macos)
- [Prepare and verify](#prepare-and-verify)
- [Run the server](#run-the-server)
- [Additional Project 1 commands](#additional-project-1-commands)
- [Model definitions](#model-definitions)
- [Data provenance](#data-provenance)
- [Report](#report)

---

## What is included

- **Task 1** – Home page listing the group members with their names and matriculation numbers (rendered from Python, not hard-coded in HTML).
- **Task 2** – Dedicated `project1` Django app, registered in `INSTALLED_APPS`, with its own `urls.py`, and a link added to the home page project list.
- **Task 3** – CSV upload, parsing, and data visualization:
  - Scatter plots of two selected features, coloured by class (classification) or one feature vs. the target (regression).
  - Automatic or user-specified detection of the problem type (classification / regression).
- **Task 4** – End-to-end supervised learning pipeline:
  - Choice of ML model (scikit-learn).
  - Train / test split.
  - Hyperparameter search over a user-defined grid.
  - Evaluation with a user-selected score.
  - Results table and best-model summary.
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
Select `env\Scripts\python.exe through Python:´Select Interpreter in VS Code.

### Linux / macOS
```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
---

## Prepare and verify
### Windows
```powershell
.\env\Scripts\python.exe manage.py migrate
.\env\Scripts\python.exe manage.py check
.\env\Scripts\python.exe manage.py makemigrations --check --dry-run
.\env\Scripts\python.exe manage.py test project1
```
### Linux/macOS
```bash
python manage.py migrate
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py test project1
```

---

## Run the server
### Windows
```powershell
.\env\Scripts\python.exe manage.py runserver
```

### Linux / macOS
```bash
python manage.py runserver
```

Open http://127.0.0.1:8000/project1/.

---
## Model definitions
### Data loading
Uploaded CSV files follow the convention:

- First row – feature names.

- Last column – target (label).

- Any id-like column is filtered out.

The problem type is either selected by the user or detected automatically:

- Classification – target has a small number of discrete values.

- Regression – target is continuous.

### Visualization
- Classification – scatter plot of two user-selected features, coloured by class.

- Regression – scatter plot of one or two features against the target.

### Training pipeline
For each selected model and each hyperparameter combination:

1)Split the dataset into train / test sets.

2)Fit the model on the training set.

3)Evaluate on the test set with the chosen score.

The best model is reported by the selected score.

### Supported models
Implemented via scikit-learn, e.g.:

- Logistic Regression

- Decision Tree

- Random Forest

- Support Vector Machine

- k-Nearest Neighbours

- Linear / Ridge Regression

### Hyperparameter search
The user selects which hyperparameters to vary and over which values (grid search over the user-specified grid).

### Scores
```markdown
Chosen by the user, e.g. accuracy, F1, precision/recall, or MSE / R² for regression.
```
--- 
## Data provenance
The interface works with any user-uploaded CSV in the standard format (features first, target last). Example datasets such as Iris can be used directly:

---
## Report
A downloadable methods and results report is available at:

```text
/project1/report/
```
The report is linked from every page of the Project 1 interface.

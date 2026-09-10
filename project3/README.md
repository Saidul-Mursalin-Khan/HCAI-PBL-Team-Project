# Project 3 - Active Learning for Learning-to-Defer

*Human-Centric Artificial Intelligence – Team Project*

This Django app implements the HCAI Project 3 active learning for learning-to-defer assignment within the team project.

---

## Table of Contents

- [What is included](#what-is-included)
- [Setup](#setup)
  - [Windows](#windows)
  - [Linux / macOS](#linux--macos)
- [Prepare ](#prepare)
- [Additional Project 3 commands](#additional-project-3-commands)
- [Run the server](#run-the-server)
- [Model definitions](#model-definitions)
- [Data provenance](#data-provenance)
- [Report](#report)

---

## What is included

- **Task 1** – AG News data loading and a TF-IDF + Logistic Regression baseline classifier.
- **Task 2** – Two simulated experts: a class-conditional **Specialist** (strong on World/Sports, weak on Business/Sci-Tech) and a uniform **Generalist** control.
- **Task 3** – A learned, threshold-calibrated deferral gate over the classifier's confidence, margin, and entropy.
- **Task 4** – Four active learning acquisition strategies (random, uncertainty, margin, hybrid) compared across a range of expert-query budgets.
- **Task 5 (optional)** – An interactive demo: paste a headline, see the classifier's prediction and the system's live deferral decision.
- Downloadable methods and results report, linked from every page.
- CSRF-protected server-side validation and experiment logging.
- Unit, integration, CSRF, and end-to-end tests.

---

## Setup

### Windows

From the repository root, create a virtual environment and install the shared requirements:

```powershell
py -m venv env
.\env\Scripts\python.exe -m pip install --upgrade pip
.\env\Scripts\python.exe -m pip install -r requirements.txt
```

Select `env\Scripts\python.exe` through **Python: Select Interpreter** in VS Code.

### Linux / macOS
From the repository root, create a virtual environment and install the shared requirements:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
If you use VS Code, select env/bin/python` through Python: Select Interpreter.


## Prepare
### Windows

```powershell
.\env\Scripts\python.exe manage.py migrate
```
### Linux/macOS

```bash
python manage.py migrate
```
##  Additional Project 3 commands 
Run these after setup, migration, and verification as needed.
### Windows
```powershell
.\env\Scripts\python.exe manage.py run_expert_sim
.\env\Scripts\python.exe manage.py run_deferral
.\env\Scripts\python.exe manage.py run_active_learning
```
### Linux/macos

```bash
python manage.py run_expert_sim
python manage.py run_deferral
python manage.py run_active_learning
```
## Run
### Windows
```powershell
.\env\Scripts\python.exe manage.py runserver

```
### Linux / macOS
```bash
python manage.py runserver

```
 Open <http://127.0.0.1:8000/project3/>.
---
## Model definitions


### Deferral gate features
 per example, from the classifier's predicted probability distribution:
 -  top-1 confidence, 
 - margin (top-1 minus top-2 probability), 
 - entropy
 
The gate is trained to predict defer = 1 exactly when the AI's own prediction is wrong and the expert's prediction is right, and its decision threshold is calibrated on held-out data to directly maximise combined system accuracy.
 
Active learning acquisition strategies:

```text
random       -- uniform random sampling (control)
uncertainty  -- lowest classifier top-1 confidence first
margin       -- smallest gap between the classifier's top-2 class probabilities
hybrid       -- half random, half uncertainty
```
The hybrid strategy avoids an "always defer" failure mode that pure uncertainty/margin sampling can produce at very small query budgets (see the report for details).

---

## Data provenance


```markdown
AG News dataset (`fancyzhx/ag_news` on Hugging Face), as specified in the assignment brief: **120,000 training articles / 7,600 test articles**, evenly split across 4 topic classes (**World**, **Sports**, **Business**, **Sci/Tech**).
https://huggingface.co/datasets/fancyzhx/ag_news
```


---
## Report

A downloadable methods and results report is available at:
```text
/project3/report/
```

The report is linked from every page of the Project 3 interface.
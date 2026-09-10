# Human-Centric Artificial Intelligence — PBL (SoSe 2026)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Framework-Django-092E20.svg)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/status-academic--project-lightgrey.svg)]()

A collection of interactive, human-centered machine learning applications developed for the **Human-Centric Artificial Intelligence (HCAI)** Project-Based Learning course at **Hamburg University of Technology (TUHH)**.

---

## Table of Contents

- [Team Members](#team-members)
- [Course Information](#course-information)
- [Summary](#summary)
- [Projects Overview](#projects-overview)
  - [01 — Automated Machine Learning](#project-01-automated-machine-learning-automl)
  - [02 — Explainability](#project-02-explainability)
  - [03 — Active Learning for Learning-to-Defer](#project-03-active-learning-for-learning-to-defer)
  - [04 — Preference Elicitation](#project-04-Preference-elicitation)
- [Technologies Used](#technologies-used)
- [Key Concepts Covered](#key-human-centric-ai-concepts-covered)
- [Prerequisites](#prerequisites)
- [Setup & Run Locally](#setup-and-run-project-locally)

---

## Team Members

- Asmaul Husna Urme
- Saidul Mursalin Khan
- Ridowana Tabassum
- Doga Ruken Günes
- Md Mehrabul Islam Zeshan

## Course Information

| | |
|---|---|
| **Course** | Human-Centric Artificial Intelligence |
| **Semester** | Summer Semester 2026 (SoSe2026) |
| **Institution** | E-EXK7 Human-Centric Machine Learning — Hamburg University of Technology (TUHH) |
| **Course Type** | Project-Based Learning (PBL) |

## Summary

This repository contains a collection of interactive machine learning projects developed for the **Human-Centric Artificial Intelligence (HCAI)** course at **Hamburg University of Technology (TUHH)**.

The projects demonstrate different human-centered AI and machine learning concepts through interactive web-based applications. They explore how users can interact with, understand, and collaborate with AI systems, covering **automated machine learning, explainable AI, active learning, and human-AI collaboration**.

The applications are built using **Django** and Python-based machine learning libraries. The projects focus not only on predictive performance, but also on **transparency, interpretability, human involvement, and the interaction between humans and AI systems**.

Overall, the repository demonstrates how machine learning systems can be designed to become more understandable, adaptable, and useful through meaningful human interaction.

---

## Projects Overview

### Project 01: Automated Machine Learning (AutoML)

An interactive supervised learning application that guides users through the main stages of a machine learning workflow, from dataset upload and visualization to model training and evaluation.

**Features**

- Upload and process structured CSV datasets
- Explore datasets through feature visualizations
- Select features and target variables
- Split data into training and testing sets
- Select and train machine learning models
- Configure relevant model hyperparameters
- Evaluate trained models using appropriate performance metrics
- Explore the different stages of an end-to-end supervised learning workflow



---

### Project 02: Explainability

An interactive Explainable AI (XAI) application based on the **Palmer Penguins dataset**. The application explores model complexity, regularization, counterfactual explanations, and global feature-effect visualization for classification models.

**Model Interpretability and Complexity**

- Train and visualize Decision Tree classifiers
- Display test accuracy and model complexity using the number of leaves
- Train models with different levels of regularization
- Use a λ slider to explore the trade-off between predictive performance and model complexity
- Train Logistic Regression models with an appropriate complexity measure
- Compare model performance and interpretability across different model configurations

**Counterfactual Explanations**

The application generates counterfactual examples that show how changing input features can lead to a desired target prediction.

Users can:

- Select an example from the dataset
- Select a desired target class
- Generate counterfactual examples
- Compare the original example with generated alternatives
- Explore counterfactuals based on the selected model type and λ value

**Feature Effect Visualization**

The application provides global model-agnostic feature-effect visualizations:

- Partial Dependence Plots (PDP)
- Accumulated Local Effects (ALE)
- Select any of the four numerical features
- Visualize the effect of the selected feature on the predicted probability of each penguin species


---

### Project 03: Active Learning for Learning-to-Defer

A human-in-the-loop machine learning system based on the **AG News dataset**, combining classification, simulated expert feedback, learning-to-defer, and active learning.

**Features**

- Train a baseline classifier for news-topic classification
- Simulate one or more non-perfect experts with different areas of expertise
- Evaluate the performance and strengths of the simulated expert(s)
- Implement a learning-to-defer strategy that chooses between the model and the expert
- Evaluate both predictive performance and the quality of deferral decisions
- Apply active learning to select informative samples for expert queries
- Investigate how expert feedback can be used to learn when deferral is beneficial

---

### Project 04: Preference Elicitation

A movie recommendation preference-elicitation system designed to study how efficiently a new user's preferences can be learned from a limited number of interactions.

The system models a user's latent preference vector using movie features extracted from the **IMDb 5000 Movie Dataset**.

**Features**

- Extract a compact feature representation from approximately 5,000 movies
- Model pairwise preferences using the Bradley-Terry model
- Extend the preference model to rankings using the Plackett-Luce formulation
- Compare two preference-elicitation interfaces:
  - Pairwise movie comparison
  - Ranking a set of ten movies
- Design a counterbalanced within-subject user study
- Evaluate preference models using held-out validation trials
- Collect workload, confidence, ease-of-use, and preference feedback
- Provide a landing page with access to the study documentation and interactive study interface

---

## Technologies Used

- **Python**
- **Django**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **SciPy**
- **Matplotlib**
- **Seaborn**
- **Hugging Face Datasets**
- **HTML5**
- **CSS3**
- **Git / GitHub**

### Datasets

- **Palmer Penguins Dataset**
- **IMDb 5000 Movie Dataset**
- **AG News Dataset**

## Key Human-Centric AI Concepts Covered

- Automated Machine Learning (AutoML)
- Supervised Learning Pipelines
- Data Visualization for Machine Learning
- Explainable Artificial Intelligence (XAI)
- Model Interpretability
- Model Complexity and Regularization
- Counterfactual Explanations
- Partial Dependence Plots (PDP)
- Accumulated Local Effects (ALE)
- Active Learning
- Simulated Expert Feedback
- Learning-to-Defer
- Human-in-the-Loop Machine Learning
- Human-AI Collaboration
- Preference Elicitation
- Recommender Systems
- Bradley-Terry Preference Modeling
- Plackett-Luce Ranking Models
- Experimental Design for Human Studies
- Human Preference Modeling

---

## Prerequisites

Before running the project, ensure the following are installed:

- **Python 3.11+**
- **Git** for cloning the repository
- **pip** for installing Python dependencies
- **Python virtual environment (`venv`)**

---

## Setup and Run Project Locally

### 1. Clone the repository

    git clone https://github.com/Saidul-Mursalin-Khan/HCAI-PBL-Team-Project.git
    cd HCAI-PBL-Team-Project

### 2. Create and activate a virtual environment

#### Windows PowerShell

    python -m venv venv
    .\venv\Scripts\Activate.ps1

#### macOS / Linux

    python3 -m venv venv
    source venv/bin/activate

### 3. Install dependencies

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

### 4. Apply Django migrations

    python manage.py migrate

### 5. Run Project 3 commands

The following commands run the main Project 3 components:

    python manage.py run_expert_sim
    python manage.py run_deferral
    python manage.py run_active_learning

- `run_expert_sim`: Simulates expert(s) and reports their accuracy on the AG News test set.
- `run_deferral`: Trains and evaluates the learning-to-defer system.
- `run_active_learning`: Runs active learning for expert-competence discovery.

### 6. Start the development server

    python manage.py runserver

The server will start at:

http://127.0.0.1:8000/

---

<p align="center"><i>Developed as part of the HCAI PBL course, TUHH — SoSe 2026</i></p>

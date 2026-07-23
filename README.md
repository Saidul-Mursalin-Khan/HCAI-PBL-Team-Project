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

This repository contains a collection of projects developed for the **Human-Centric Artificial Intelligence (HCAI)** course at **TUHH**.

The projects demonstrate the application of human-centered AI and machine learning concepts through interactive web-based interfaces. Each project focuses on how humans can interact with, understand, and influence AI systems — covering automated machine learning, explainable AI, active learning, and human-AI collaboration.

All applications are built with **Django**, providing hands-on experience with real-world machine learning workflows while keeping a strong focus on transparency, interpretability, and effective human involvement. The goal is not only to build accurate models, but to investigate how AI systems can become more understandable, adaptable, and reliable through human interaction.

---

## Projects Overview

### Project 01: Automated Machine Learning (AutoML)

An interactive supervised learning interface for exploring the complete ML pipeline — from loading a dataset to training and evaluating models.

**Features**
- Upload and process structured (CSV) datasets
- Visualize feature relationships and data distributions
- Select and train machine learning models
- Split datasets into training and testing sets
- Evaluate model performance with suitable metrics
- Explore each stage of a typical ML workflow



---

### Project 02: Explainability

An interactive explainable AI (XAI) application built on the **Palmer Penguins dataset**, exploring model behavior and the trade-off between complexity, accuracy, and interpretability.

**Model Interpretability**
- Train decision tree models at different complexity levels
- Visualize decision trees and evaluate performance
- Control model sparsity through regularization
- Train logistic regression models with complexity measures
- Compare accuracy vs. interpretability trade-offs

**Counterfactual Explanations**

The system generates personalized counterfactual examples showing how predictions change with different input features. Users can:
- Select an instance from the dataset
- Choose a desired target prediction
- Generate alternative examples leading to that outcome
- Identify which features drive model decisions

**Feature Effect Visualization**
- Partial Dependence Plots (PDP)
- Accumulated Local Effects (ALE) plots


---

### Project 03: Active Learning for Learning-to-Defer

A human-in-the-loop system combining **active learning** with **learning-to-defer**, investigating how AI can recognize uncertain situations and choose to predict independently or request human input.

**Features**
- Train classification models
- Apply active learning strategies to select informative samples
- Simulate expert feedback for challenging cases
- Implement learning-to-defer mechanisms for human-AI collaboration
- Compare traditional ML approaches against human-augmented systems
- Evaluate how expert intervention affects performance and reliability

---

## Technologies Used

- Python
- Django Framework
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Machine Learning Models
- Data Visualization Techniques
- Interactive Web Interfaces
- Human-in-the-Loop Machine Learning Approaches

## Key Human-Centric AI Concepts Covered

- Automated Machine Learning (AutoML)
- Supervised Learning Pipelines
- Data Visualization for Machine Learning
- Explainable Artificial Intelligence (XAI)
- Model Interpretability
- Model Complexity and Regularization
- Counterfactual Explanations
- Feature Effect Visualization (PDP and ALE)
- Active Learning
- Learning-to-Defer
- Human-in-the-Loop Machine Learning
- AI-Assisted Decision Making
- Trustworthy and Collaborative AI Systems

---

## Prerequisites

Before running this project collection, ensure the following are installed:

- **Python 3.8+** (3.11+ recommended)
- **Git** for cloning the repository
- **pip** for installing Python dependencies
- **Virtual environment tools** (`venv`, included with Python 3.3+)

---

## Setup and Run Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Saidul-Mursalin-Khan/HCAI-PBL-Team-Project.git
cd HCAI-PBL-Team-Project
```

### 2. Create and activate a virtual environment

```bash
# Create
python3 -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### 4. Set up the Django application

```bash
python3 manage.py migrate
```

### 5. Run additional commands for Project 3

```bash
python manage.py run_expert_sim
python manage.py run_deferral
python manage.py run_active_learning
```

### 6. Run the development server

```bash
python3 manage.py runserver
```

The server will start at **http://127.0.0.1:8000/**

---

<p align="center"><i>Developed as part of the HCAI PBL course, TUHH — SoSe 2026</i></p>

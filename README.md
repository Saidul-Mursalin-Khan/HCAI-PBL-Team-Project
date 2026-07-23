# Human-Centric Artificial Intelligence PBL (SoSe2026)

# Course Information

Course: Human-Centric Artificial Intelligence
Semester: Summer Semester 2026
Institution: E-EXK7 Human-Centric Machine Learning - Hamburg University of Technology (TUHH)
Course Type: Project-Based Learning (PBL)

# Team Members






## Summary

This repository contains a collection of projects developed for the **Human-Centric Artificial Intelligence (HCAI)** course at the **Hamburg University of Technology (TUHH)**.

The projects demonstrate the application of human-centered AI and machine learning concepts through interactive web-based interfaces. Each project focuses on how humans can interact with, understand, and influence AI systems.

The implemented projects cover topics including automated machine learning, explainable AI, active learning, and human-AI collaboration. The applications are designed to provide hands-on experience with real-world machine learning workflows while maintaining a focus on transparency, interpretability, and effective human involvement.

All projects are implemented using the **Django framework**, providing interactive user interfaces that allow users to explore different machine learning techniques and understand the role of humans in AI decision-making.

The goal of these projects is not only to build accurate machine learning models but also to investigate how AI systems can become more understandable, adaptable, and reliable through human interaction.

---

# Projects Overview

## Project 01: Automated Machine Learning (AutoML)

An interactive supervised learning interface that allows users to explore the complete machine learning pipeline, from loading datasets to training and evaluating machine learning models.

The application enables users to upload CSV datasets, visualize data characteristics, and train different machine learning models through an easy-to-use interface.

The project includes:

- Uploading and processing structured datasets.
- Visualizing feature relationships and data distributions.
- Selecting and training machine learning models.
- Splitting datasets into training and testing sets.
- Evaluating model performance using suitable metrics.
- Exploring the different stages of a typical machine learning workflow.

This project demonstrates how automated machine learning tools can simplify model development while keeping users involved in important decisions during the ML process.

---

## Project 02: Explainability

An interactive explainable AI application based on the **Palmer Penguins dataset** that allows users to explore machine learning model behavior and understand AI predictions.

The project focuses on improving model transparency by investigating the relationship between model complexity, accuracy, and interpretability.

The application includes:

### Model Interpretability

- Training decision tree models with different complexity levels.
- Visualizing decision trees and evaluating their performance.
- Allowing users to control model sparsity through regularization.
- Training logistic regression models with suitable complexity measures.
- Comparing model accuracy and interpretability trade-offs.

### Counterfactual Explanations

The system generates personalized counterfactual examples to explain how predictions can change based on modifications to input features.

Users can:

- Select an instance from the dataset.
- Choose a desired target prediction.
- Generate alternative examples that lead to the selected outcome.
- Understand which features influence model decisions.

### Feature Effect Visualization

The application provides global explanations using:

- Partial Dependence Plots (PDP)
- Accumulated Local Effects (ALE) plots

These visualizations help users understand how individual features affect model predictions.

This project demonstrates how explainable AI techniques can improve trust and understanding of machine learning systems.

---

## Project 03: Active Learning for Learning-to-Defer

A human-in-the-loop machine learning system that combines **active learning** with **learning-to-defer** approaches to improve AI decision-making.

The project investigates how AI systems can recognize uncertain situations and decide whether to make predictions independently or request assistance from a human expert.

The application includes:

- Training classification models using machine learning techniques.
- Applying active learning strategies to select informative samples.
- Simulating expert feedback for challenging cases.
- Implementing learning-to-defer mechanisms for human-AI collaboration.
- Comparing traditional machine learning approaches with systems that incorporate human expertise.
- Evaluating how expert intervention affects prediction performance and reliability.

This project highlights the importance of trustworthy AI systems where models understand their limitations and collaborate effectively with humans.

---




# Technologies Used

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

---

# Key Human-Centric AI Concepts Covered

The projects explore the following concepts:

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









# Prerequisites

Before running this project collection, ensure that the following software is installed:

- **Python 3.8 or higher** (Python 3.11+ recommended)
- **Git** for cloning the repository
- **pip** for installing Python dependencies
- **Virtual environment tools** (`venv` module is included with Python 3.3+)

---

# Setup and Run Project Locally

# Step 1: Clone the Repository
`git clone <https://github.com/Saidul-Mursalin-Khan/HCAI-PBL-Team-Project.git>`

# Step 2: Create and Activate Virtual Environment
# Create virtual environment
`python3 -m venv venv`

# Activate virtual environment
# On macOS/Linux:
`source venv/bin/activate`

# On Windows:
# `venv\Scripts\activate`


# Step 3: Install Dependencies
# Ensure pip is up to date
`pip3 install --upgrade pip`

# Install project dependencies
`pip3 install -r requirements.txt`


# Step 4: Set Up and Run Django Application
# Run database migrations
`python3 manage.py migrate`


# Step 5: Additional command for project 3
 `python manage.py run_expert_sim`
 `python manage.py run_deferral`
 `python manage.py run_active_learning`

 # Step 6: Run the Development Server
 `python3 manage.py runserver`
 `python manage.py runserver`

# `The server will start at http://127.0.0.1:8000/`

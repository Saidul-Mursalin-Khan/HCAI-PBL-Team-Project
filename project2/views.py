from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from palmerpenguins import load_penguins
import numpy as np


def index(request):
    # Default lambda value
    lambda_value = 0.1

    # If user submitted the form, get lambda from POST
    if request.method == "POST":
        lambda_value = float(request.POST.get("lambda_value", 0.1))

    # Load Palmer Penguins dataset
    df = load_penguins()
    df = df.dropna()  # remove missing values

    # Select features and target
    X = df[['bill_length_mm', 'bill_depth_mm',
            'flipper_length_mm', 'body_mass_g']]
    y = df['species']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define possible max leaves and lambda
    max_leaves_options = [3, 5, 7, 10, 14, 20]

    best_score = -1
    best_tree = None
    best_leaves = None

    # Choose the best tree according to acctest + lambda * n_leaves
    for leaves in max_leaves_options:
        tree = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=42)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        acctest = accuracy_score(y_test, y_pred)
        score = acctest + lambda_value * leaves

        if score > best_score:
            best_score = score
            best_tree = tree
            best_leaves = leaves

    # Prepare context for template
    context = {
        'title': 'Project 2: Decision Tree with Regularization',
        'accuracy': accuracy_score(y_test, best_tree.predict(X_test)),
        'n_leaves': best_leaves,
        'lambda_value': lambda_value,
    }

    # Train a decision tree
    # tree = DecisionTreeClassifier(random_state=42)
    # tree.fit(X_train, y_train)

    # Predict on test set and calculate accuracy
    # y_pred = tree.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)

    # Number of leaves in the tree
    # n_leaves = tree.get_n_leaves()

    # context = {
    #     'title': 'Project 2: Decision Tree for Palmer Penguins',
    #     'accuracy': accuracy,
    #     'n_leaves': n_leaves,
    # }

    return render(request, 'project2/index.html', context)


def logistic_view(request):
    df = load_penguins()
    df = df.dropna()

    X = df[['bill_length_mm', 'bill_depth_mm',
            'flipper_length_mm', 'body_mass_g']]
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train logistic regression with L2 regularization (C=1/lambda)
    lambda_value = 0.1
    if request.method == "POST":
        lambda_value = float(request.POST.get("lambda_value", 0.1))

    logreg = LogisticRegression(penalty="l2", C=1/lambda_value, max_iter=1000)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    complexity = np.sum(np.abs(logreg.coef_))  # L1 norm of coefficients

    context = {
        'title': 'Project 2: Logistic Regression',
        'accuracy': accuracy,
        'complexity': complexity,
        'lambda_value': lambda_value,
    }

    return render(request, 'project2/logistic.html', context)

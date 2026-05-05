
import os
import uuid
import pickle

import numpy as np
import pandas as pd
from .models import UploadedDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.shortcuts import render, redirect
from django.http import HttpResponse

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error, mean_absolute_percentage_error,
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    roc_curve
)

# =====================
# SETTINGS
# =====================
MEDIA_ROOT = 'media/plots/'
SPLIT_DIR = 'media/splits/'

# =====================
# UTILITIES
# =====================

def save_fig(fig, name):
    os.makedirs(MEDIA_ROOT, exist_ok=True)
    filename = f"{name}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(MEDIA_ROOT, filename)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return '/' + path


def detect_problem_type(series):
    if series.dtype == 'object' or series.nunique() <= 20:
        return 'classification'
    return 'regression'

# =====================
# HOME
# =====================

def index(request):
    return render(request, 'project1/index.html', {
        'title': 'Project 1: Supervised Learning Interface'
    })

# =====================
# UPLOAD
# =====================

def upload(request):
    error = None

    if request.method == 'POST' and request.FILES.get('dataset'):
        uploaded_file = request.FILES['dataset']

        if not uploaded_file.name.endswith('.csv'):
            error = "Only CSV files are supported."
        else:
            obj = UploadedDataset(file=uploaded_file)
            obj.save()

            file_path = obj.file.path
            df = pd.read_csv(file_path)

            request.session['dataset_path'] = file_path
            request.session['columns'] = list(df.columns)

            return render(request, 'project1/upload.html', {
                'table_html': df.head(20).to_html(classes='data-table', index=False),
                'columns': list(df.columns),
                'shape': df.shape,
            })

    return render(request, 'project1/upload.html', {'error': error})

# =====================
# VISUALIZATION
# =====================

def visualize(request):
    path = request.session.get('dataset_path')
    if not path:
        return redirect('project1:upload')

    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    target = request.session.get('target_column', df.columns[-1])

    plots = {}

    # Histogram
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 4))
    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        df[col].hist(ax=ax, bins=20, color='steelblue', edgecolor='white')
        ax.set_title(col)

    plots['histogram'] = save_fig(fig, 'histogram')

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    df[numeric_cols].boxplot(ax=ax)
    plots['boxplot'] = save_fig(fig, 'boxplot')

    # Scatter
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots()
        ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
        plots['scatter'] = save_fig(fig, 'scatter')

    return render(request, 'project1/visualize.html', {'plots': plots})

# =====================
# MODEL MAPS
# =====================

REG_MODELS = {
    'linear_regression': LinearRegression,
    'decision_tree': DecisionTreeRegressor,
    'random_forest': RandomForestRegressor,
    'knn': KNeighborsRegressor,
    'svr': SVR,
}

REG_PARAMS = {
    'decision_tree': {'max_depth': int},
    'random_forest': {'n_estimators': int, 'max_depth': int},
    'knn': {'n_neighbors': int},
    'svr': {'C': float, 'kernel': str},
}

CLS_MODELS = {
    'logistic_regression': LogisticRegression,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'knn': KNeighborsClassifier,
    'naive_bayes': GaussianNB,
    'svm': SVC,
}

CLS_PARAMS = {
    'logistic_regression': {'C': float},
    'decision_tree': {'max_depth': int},
    'random_forest': {'n_estimators': int, 'max_depth': int},
    'knn': {'n_neighbors': int},
    'svm': {'C': float, 'kernel': str},
}

# =====================
# CONFIGURE
# =====================

def configure(request):
    path = request.session.get('dataset_path')
    if not path:
        return redirect('project1:upload')

    df = pd.read_csv(path)

    if request.method == 'POST':
        target = request.POST.get('target_column')
        problem_type = request.POST.get('problem_type')
        test_size = float(request.POST.get('test_size', 0.2))

        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        os.makedirs(SPLIT_DIR, exist_ok=True)
        split_id = uuid.uuid4().hex[:8]
        split_path = os.path.join(SPLIT_DIR, f"split_{split_id}.pkl")

        with open(split_path, 'wb') as f:
            pickle.dump({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
            }, f)

        request.session['target_column'] = target
        request.session['problem_type'] = problem_type or detect_problem_type(y)
        request.session['split_path'] = split_path

        if request.session['problem_type'] == 'classification':
            return redirect('project1:classification')
        return redirect('project1:regression')

    return render(request, 'project1/configure.html', {
        'columns': df.columns,
    })

# =====================
# REGRESSION TRAIN
# =====================

def regression_train(request):
    if request.session.get('problem_type') != 'regression':
        return redirect('project1:configure')

    with open(request.session['split_path'], 'rb') as f:
        split = pickle.load(f)

    X_train, X_test = split['X_train'], split['X_test']
    y_train, y_test = split['y_train'], split['y_test']

    results = None

    if request.method == 'POST':
        model_key = request.POST.get('model')
        params = {}

        Model = REG_MODELS[model_key]
        model = Model(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results = {
            'r2': r2_score(y_test, pred),
            'mae': mean_absolute_error(y_test, pred),
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        }

    return render(request, 'project1/regression.html', {
        'models': REG_MODELS,
        'results': results
    })

# =====================
# CLASSIFICATION TRAIN
# =====================

def classification_train(request):
    if request.session.get('problem_type') != 'classification':
        return redirect('project1:configure')

    with open(request.session['split_path'], 'rb') as f:
        split = pickle.load(f)

    X_train, X_test = split['X_train'], split['X_test']
    y_train, y_test = split['y_train'], split['y_test']

    results = None

    if request.method == 'POST':
        model_key = request.POST.get('model')
        Model = CLS_MODELS[model_key]

        model = Model()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results = {
            'accuracy': accuracy_score(y_test, pred),
            'f1': f1_score(y_test, pred, average='weighted'),
        }

    return render(request, 'project1/classification.html', {
        'models': CLS_MODELS,
        'results': results
    })

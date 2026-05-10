import os
import uuid
import pickle
import pandas as pd
from django.shortcuts import render, redirect
from sklearn.model_selection import train_test_split
import json

SPLIT_DIR = 'media/splits/'

def detect_problem_type(series):
    """Auto-detect: if target has ≤ 20 unique values or is object → classification."""
    if series.dtype == 'object' or series.nunique() <= 20:
        return 'classification'
    return 'regression'

def configure(request):
    dataset_path = request.session.get('dataset_path')
    if not dataset_path:
        return redirect('project1:upload')

    df = pd.read_csv(dataset_path)

    # Receive target column from upload form POST or session
    if request.method == 'POST':
        target_col = request.POST.get('target_column') or request.session.get('target_column')
        problem_type = request.POST.get('problem_type')  # user can override
        test_size = float(request.POST.get('test_size', 0.2))

        if not target_col or target_col not in df.columns:
            return render(request, 'project1/configure.html', {
                'error': 'Invalid target column.',
                'columns': df.columns,
            })

        # Auto-detect if not manually set
        if not problem_type:
            problem_type = detect_problem_type(df[target_col])

        # Split
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical features for model training
        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Save split to disk
        os.makedirs(SPLIT_DIR, exist_ok=True)
        split_id = uuid.uuid4().hex[:8]
        split_path = os.path.join(SPLIT_DIR, f'split_{split_id}.pkl')
        with open(split_path, 'wb') as f:
            pickle.dump({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': list(X.columns),
            }, f)

        # Store in session
        request.session['target_column'] = target_col
        request.session['problem_type'] = problem_type
        request.session['test_size'] = test_size
        request.session['split_path'] = split_path

        # Redirect to correct training view
        return redirect('project1:visualize')

    # GET — show config form
    # GET — show config form
    target_col = request.session.get('target_column')
    if not target_col or target_col not in df.columns:
        target_col = df.columns[-1]

    saved_test_size = request.session.get('test_size', 0.2)
    saved_test_pct = int(saved_test_size * 100)

    auto_type = detect_problem_type(df[target_col])

    column_types = {}

    for col in df.columns:
        column_types[col] = detect_problem_type(df[col])

    return render(request, 'project1/configure.html', {
        'columns': df.columns,
        'target_column': target_col,
        'auto_detected_type': auto_type,
        'column_types_json': json.dumps(column_types),
        'test_size': saved_test_size,
        'test_size_pct': saved_test_pct,
    })
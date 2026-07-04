import os
import uuid
import pickle
from django.shortcuts import render, redirect
from sklearn.model_selection import train_test_split

from .common import (
    build_config_context,
    get_splits_dir,
    parse_test_size,
    prepare_features,
    read_csv_dataset,
    validate_problem_type,
)

def configure(request):
    dataset_path = request.session.get('dataset_path')
    if not dataset_path:
        return redirect('project1:upload')

    try:
        df = read_csv_dataset(dataset_path)
    except ValueError as exc:
        return render(request, 'project1/upload.html', {'error': str(exc)})

    # Receive target column from upload form POST or session
    if request.method == 'POST':
        target_col = request.POST.get('target_column') or request.session.get('target_column')
        problem_type = request.POST.get('problem_type')  # user can override
        try:
            test_size = parse_test_size(request.POST.get('test_size', 0.2))
        except ValueError as exc:
            return render(request, 'project1/configure.html', build_config_context(
                df, error=str(exc), target_column=target_col
            ))

        if not target_col or target_col not in df.columns:
            return render(request, 'project1/configure.html', build_config_context(
                df, error='Invalid target column.'
            ))

        try:
            problem_type = validate_problem_type(problem_type, df[target_col])
        except ValueError as exc:
            return render(request, 'project1/configure.html', build_config_context(
                df, error=str(exc), target_column=target_col, test_size=test_size
            ))

        # Split
        prepared_df = df.dropna(subset=[target_col])
        if len(prepared_df) < 2:
            return render(request, 'project1/configure.html', build_config_context(
                df,
                error='The target column must contain at least two non-empty rows.',
                target_column=target_col,
                test_size=test_size,
                problem_type=problem_type,
            ))

        X = prepared_df.drop(columns=[target_col])
        y = prepared_df[target_col]

        if X.shape[1] == 0:
            return render(request, 'project1/configure.html', build_config_context(
                df,
                error='Select a target that leaves at least one feature column.',
                target_column=target_col,
                test_size=test_size,
                problem_type=problem_type,
            ))

        if problem_type == 'classification' and y.nunique() < 2:
            return render(request, 'project1/configure.html', build_config_context(
                df,
                error='Classification needs at least two target classes.',
                target_column=target_col,
                test_size=test_size,
                problem_type=problem_type,
            ))

        stratify = None
        if problem_type == 'classification':
            class_counts = y.value_counts()
            test_rows = int(round(len(y) * test_size))
            train_rows = len(y) - test_rows
            if class_counts.min() >= 2 and test_rows >= y.nunique() and train_rows >= y.nunique():
                stratify = y

        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )
            X_train, X_test, feature_names = prepare_features(X_train_raw, X_test_raw)
        except ValueError as exc:
            return render(request, 'project1/configure.html', build_config_context(
                df,
                error=f'Could not prepare this dataset: {exc}',
                target_column=target_col,
                test_size=test_size,
                problem_type=problem_type,
            ))

        # Save split to disk
        splits_dir = get_splits_dir()
        os.makedirs(splits_dir, exist_ok=True)
        split_id = uuid.uuid4().hex[:8]
        split_path = os.path.join(splits_dir, f'split_{split_id}.pkl')
        with open(split_path, 'wb') as f:
            pickle.dump({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': feature_names,
            }, f)

        # Store in session
        request.session['target_column'] = target_col
        request.session['problem_type'] = problem_type
        request.session['test_size'] = test_size
        request.session['split_path'] = split_path

        # Redirect to correct training view
        return redirect('project1:visualize')

    target_col = request.session.get('target_column')
    if not target_col or target_col not in df.columns:
        target_col = df.columns[-1]

    saved_test_size = request.session.get('test_size', 0.2)

    return render(request, 'project1/configure.html', build_config_context(
        df, target_column=target_col, test_size=saved_test_size
    ))

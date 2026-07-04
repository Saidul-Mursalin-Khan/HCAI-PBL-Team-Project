import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.shortcuts import render, redirect
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error, mean_absolute_percentage_error
)

from .common import save_plot

MODELS = {
    'linear_regression': LinearRegression,
    'decision_tree':     DecisionTreeRegressor,
    'random_forest':     RandomForestRegressor,
    'knn':               KNeighborsRegressor,
    'svr':               SVR,
}

HYPERPARAMS = {
    'linear_regression': {},
    'decision_tree':     {'max_depth': int},
    'random_forest':     {'n_estimators': int, 'max_depth': int},
    'knn':               {'n_neighbors': int},
    'svr':               {'C': float, 'kernel': str},
}

def regression_train(request):

    if request.session.get('problem_type') != 'regression':
        return redirect('project1:configure')

    split_path = request.session.get('split_path')

    if not split_path:
        return redirect('project1:configure')

    # RESET TRAINING STATUS WHEN PAGE LOADS
    if request.method == 'GET':
        request.session['training_completed'] = False

    try:
        with open(split_path, 'rb') as f:
            split = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError):
        request.session.pop('split_path', None)
        return redirect('project1:configure')

    X_train = split['X_train']
    X_test  = split['X_test']
    y_train = split['y_train']
    y_test  = split['y_test']

    results = None
    selected_model = None
    error = None

    if request.method == 'POST':

        model_key = request.POST.get('model')

        selected_model = model_key

        if model_key not in MODELS:
            return render(request, 'project1/regression.html', {
                'models': list(MODELS.keys()),
                'hyperparams': HYPERPARAMS,
                'results': None,
                'selected_model': selected_model,
                'error': 'Select a valid regression model.',
            })

        kwargs = {}

        for param, dtype in HYPERPARAMS.get(model_key, {}).items():

            val = request.POST.get(param)

            if val:
                try:
                    kwargs[param] = dtype(val)

                except ValueError:
                    pass

        try:
            ModelClass = MODELS[model_key]
            model = ModelClass(**kwargs)

            # TRAIN MODEL
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except ValueError as exc:
            error = f'Could not train this model: {exc}'
        else:
            # ONLY AFTER TRAINING
            request.session['training_completed'] = True

            r2 = r2_score(y_test, y_pred)

            n = len(y_test)

            p = X_test.shape[1]

            if n - p - 1 > 0:
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            else:
                adj_r2 = None

            mse = mean_squared_error(y_test, y_pred)

            rmse = np.sqrt(mse)

            mae = mean_absolute_error(y_test, y_pred)

            mape = mean_absolute_percentage_error(y_test, y_pred)

            results = {
                'r2': round(r2, 4),

                'adj_r2': round(adj_r2, 4) if adj_r2 is not None else 'N/A',

                'mae': round(mae, 4),

                'mse': round(mse, 4),

                'rmse': round(rmse, 4),

                'mape': round(mape * 100, 2),
            }

            # ACTUAL VS PREDICTED
            fig, ax = plt.subplots(figsize=(6, 5))

            ax.scatter(
                y_test,
                y_pred,
                alpha=0.6,
                color='steelblue'
            )

            mn = min(y_test.min(), y_pred.min())

            mx = max(y_test.max(), y_pred.max())

            ax.plot(
                [mn, mx],
                [mn, mx],
                'r--',
                label='Perfect fit'
            )

            ax.set_xlabel('Actual')

            ax.set_ylabel('Predicted')

            ax.set_title('Actual vs Predicted')

            ax.legend()

            results['actual_vs_pred_img'] = save_plot(
                fig,
                'actual_vs_pred'
            )

            # RESIDUALS
            residuals = y_test - y_pred

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.scatter(
                y_pred,
                residuals,
                alpha=0.6,
                color='coral'
            )

            ax.axhline(
                0,
                color='black',
                linestyle='--'
            )

            ax.set_xlabel('Predicted')

            ax.set_ylabel('Residuals')

            ax.set_title('Residual Plot')

            results['residuals_img'] = save_plot(
                fig,
                'residuals'
            )

    return render(request, 'project1/regression.html', {
        'models': list(MODELS.keys()),
        'hyperparams': HYPERPARAMS,
        'results': results,
        'selected_model': selected_model,
        'error': error,
    })

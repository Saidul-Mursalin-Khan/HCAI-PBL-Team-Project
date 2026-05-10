import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, uuid
from django.shortcuts import render, redirect
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

MODELS = {
    'logistic_regression': LogisticRegression,
    'decision_tree':       DecisionTreeClassifier,
    'random_forest':       RandomForestClassifier,
    'knn':                 KNeighborsClassifier,
    'naive_bayes':         GaussianNB,
    'svm':                 SVC,
}

HYPERPARAMS = {
    'logistic_regression': {'C': float},
    'decision_tree':       {'max_depth': int},
    'random_forest':       {'n_estimators': int, 'max_depth': int},
    'knn':                 {'n_neighbors': int},
    'naive_bayes':         {},
    'svm':                 {'C': float, 'kernel': str},
}

MEDIA_ROOT = 'media/plots/'

def save_fig(fig, name):
    os.makedirs(MEDIA_ROOT, exist_ok=True)
    filename = f"{name}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(MEDIA_ROOT, filename)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return '/' + path

def classification_train(request):

    if request.session.get('problem_type') != 'classification':
        return redirect('project1:configure')

    split_path = request.session.get('split_path')
    if not split_path:
        return redirect('project1:configure')

    # RESET TRAINING STATE WHEN PAGE OPENS
    if request.method == 'GET':
        request.session['training_completed'] = False

    with open(split_path, 'rb') as f:
        split = pickle.load(f)

    X_train = split['X_train']
    X_test  = split['X_test']
    y_train = split['y_train']
    y_test  = split['y_test']

    results = None
    selected_model = None

    if request.method == 'POST':

        model_key = request.POST.get('model')
        selected_model = model_key

        # Parse hyperparameters
        kwargs = {}

        for param, dtype in HYPERPARAMS.get(model_key, {}).items():
            val = request.POST.get(param)

            if val:
                try:
                    kwargs[param] = dtype(val)
                except ValueError:
                    pass

        # Special case for SVM
        if model_key == 'svm':
            kwargs['probability'] = True

        ModelClass = MODELS[model_key]

        model = ModelClass(**kwargs)

        # TRAIN MODEL
        model.fit(X_train, y_train)

        # ONLY NOW MARK TRAINING COMPLETE
        request.session['training_completed'] = True

        y_pred = model.predict(X_test)

        classes = model.classes_
        is_binary = len(classes) == 2

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average='weighted',
            zero_division=0
        )
        recall = recall_score(
            y_test,
            y_pred,
            average='weighted',
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average='weighted',
            zero_division=0
        )

        # Metrics
        results = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),

            # FOR UI PERCENT DISPLAY
            'accuracy_pct': round(accuracy * 100, 2),
            'precision_pct': round(precision * 100, 2),
            'recall_pct': round(recall * 100, 2),
            'f1_pct': round(f1 * 100, 2),
        }

        # CONFUSION MATRIX
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(cm, cmap='Blues')

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)

        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)

        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha='center',
                    va='center',
                    color='black'
                )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        fig.colorbar(im)

        results['confusion_matrix_img'] = save_fig(
            fig,
            'confusion_matrix'
        )

        # ROC CURVE
        if is_binary and hasattr(model, 'predict_proba'):

            y_prob = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_prob)

            results['roc_auc'] = round(auc, 4)
            results['roc_auc_pct'] = round(auc * 100, 2)

            fpr, tpr, _ = roc_curve(
                y_test,
                y_prob,
                pos_label=classes[1]
            )

            fig, ax = plt.subplots(figsize=(6, 5))

            ax.plot(
                fpr,
                tpr,
                label=f'AUC = {auc:.3f}'
            )

            ax.plot([0, 1], [0, 1], 'k--')

            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')

            ax.set_title('ROC Curve')

            ax.legend()

            results['roc_img'] = save_fig(fig, 'roc')

    return render(request, 'project1/classification.html', {
        'models': list(MODELS.keys()),
        'hyperparams': HYPERPARAMS,
        'results': results,
        'selected_model': selected_model,
    })
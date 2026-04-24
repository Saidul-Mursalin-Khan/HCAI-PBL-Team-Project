import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — required for Django
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render, redirect

MEDIA_ROOT = 'media/plots/'

def save_fig(fig, name):
    """Save matplotlib figure to media folder, return relative URL."""
    os.makedirs(MEDIA_ROOT, exist_ok=True)
    filename = f"{name}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(MEDIA_ROOT, filename)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return '/' + path  # URL to serve

def visualize(request):
    path = request.session.get('dataset_path')
    if not path:
        return redirect('project1:upload')

    df = pd.read_csv(path)
    target = request.session.get('target_column', df.columns[-1])
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    plots = {}

    # 1. Histogram — all numeric columns
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 4))
    if len(numeric_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        df[col].hist(ax=ax, bins=20, color='steelblue', edgecolor='white')
        ax.set_title(col)
    plots['histogram'] = save_fig(fig, 'histogram')

    # 2. Box plot
    fig, ax = plt.subplots(figsize=(10, 5))
    df[numeric_cols].boxplot(ax=ax)
    ax.set_title('Box Plot')
    plots['boxplot'] = save_fig(fig, 'boxplot')

    # 3. Scatter plot — first two numeric columns
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6, c='steelblue')
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title('Scatter Plot')
        plots['scatter'] = save_fig(fig, 'scatter')

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', ax=ax, cmap='coolwarm')
    ax.set_title('Correlation Heatmap')
    plots['heatmap'] = save_fig(fig, 'heatmap')

    # 5. Count plot — target column
    if target in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[target].value_counts().plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'Count Plot — {target}')
        plots['countplot'] = save_fig(fig, 'countplot')

    return render(request, 'project1/visualize.html', {'plots': plots})
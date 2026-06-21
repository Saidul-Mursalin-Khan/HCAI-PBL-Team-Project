import json
import base64
import io
from urllib.parse import urlencode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.shortcuts import render
from palmerpenguins import load_penguins
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
MODEL_FEATURES = NUMERIC_FEATURES + ["island", "sex", "year"]
CATEGORICAL_FEATURES = ["island", "sex", "year"]
TREE_LEAF_OPTIONS = [3, 5, 7, 10, 14, 20, 30]
LOGISTIC_C_OPTIONS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def _load_data():
    df = load_penguins().dropna().reset_index(drop=True)
    df["year"] = df["year"].astype(str)
    return df


def _split_data():
    df = _load_data()
    x = df[MODEL_FEATURES]
    y = df["species"]
    return train_test_split(
        x,
        y,
        df,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def _preprocessor(scale_numeric=False):
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric_step, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def _feature_names(model):
    return list(model.named_steps["prep"].get_feature_names_out())


def _tree_candidate(max_leaf_nodes):
    return Pipeline(
        [
            ("prep", _preprocessor(scale_numeric=False)),
            (
                "model",
                DecisionTreeClassifier(
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=42,
                ),
            ),
        ]
    )


def _logistic_candidate(c_value):
    return Pipeline(
        [
            ("prep", _preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _logistic_complexity(model):
    return float(np.sum(np.abs(model.named_steps["model"].coef_)))


def _select_model(model_type, lambda_value, x_train, x_test, y_train, y_test):
    candidates = []
    if model_type == "logistic":
        for c_value in LOGISTIC_C_OPTIONS:
            model = _logistic_candidate(c_value)
            model.fit(x_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(x_test))
            complexity = _logistic_complexity(model)
            candidates.append(
                {
                    "label": f"C={c_value:g}",
                    "model": model,
                    "accuracy": accuracy,
                    "complexity": complexity,
                    "score": accuracy - lambda_value * complexity,
                    "setting": c_value,
                }
            )
    else:
        for leaves in TREE_LEAF_OPTIONS:
            model = _tree_candidate(leaves)
            model.fit(x_train, y_train)
            tree = model.named_steps["model"]
            accuracy = accuracy_score(y_test, model.predict(x_test))
            complexity = tree.get_n_leaves()
            candidates.append(
                {
                    "label": f"max leaves={leaves}",
                    "model": model,
                    "accuracy": accuracy,
                    "complexity": complexity,
                    "score": accuracy - lambda_value * complexity,
                    "setting": leaves,
                }
            )

    selected = max(candidates, key=lambda row: row["score"])
    return selected, candidates


def _model_summary(selected, model_type):
    model = selected["model"]
    if model_type == "logistic":
        classifier = model.named_steps["model"]
        names = _feature_names(model)
        rows = []
        for class_name, coef in zip(classifier.classes_, classifier.coef_):
            top = sorted(
                zip(names, coef),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:7]
            rows.append(
                {
                    "class_name": class_name,
                    "coefficients": [
                        {"feature": name, "value": float(value)} for name, value in top
                    ],
                }
            )
        return {"type": "logistic", "rows": rows}

    tree = model.named_steps["model"]
    return {
        "type": "tree",
        "tree_text": export_text(
            tree,
            feature_names=_feature_names(model),
            max_depth=5,
            decimals=2,
        ),
        "image_uri": _tree_image_uri(model),
    }


def _tree_image_uri(model):
    tree = model.named_steps["model"]
    fig_height = max(6.0, min(14.0, 1.4 + tree.get_n_leaves() * 0.35))
    fig_width = max(14.0, min(24.0, 7.5 + tree.get_depth() * 1.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    fig.patch.set_facecolor("#f4f7fb")
    ax.set_facecolor("white")
    ax.axis("off")
    plot_tree(
        tree,
        ax=ax,
        feature_names=_feature_names(model),
        class_names=[str(c) for c in tree.classes_],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=10,
        label="all",
        node_ids=True,
    )
    buffer = io.BytesIO()
    fig.tight_layout(pad=0.8)
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _mad(series):
    median = np.median(series)
    value = np.median(np.abs(series - median))
    return float(value if value > 0 else np.std(series) or 1.0)


def _display_value(feature, value):
    if feature in NUMERIC_FEATURES:
        return f"{float(value):.2f}"
    return str(value)


def _counterfactuals(model, df, example_index, target_label, sample_count=3500, limit=5):
    rng = np.random.default_rng(7)
    base = df.loc[example_index, MODEL_FEATURES].copy()
    samples = pd.DataFrame(index=range(sample_count), columns=MODEL_FEATURES)

    for feature in NUMERIC_FEATURES:
        scale = _mad(df[feature])
        values = rng.normal(float(base[feature]), scale * 0.75, sample_count)
        samples[feature] = np.clip(values, df[feature].min(), df[feature].max())

    for feature in CATEGORICAL_FEATURES:
        categories = df[feature].astype(str).unique()
        keep = rng.random(sample_count) < 0.65
        random_values = rng.choice(categories, size=sample_count)
        samples[feature] = np.where(keep, str(base[feature]), random_values)

    predictions = model.predict(samples)
    matches = samples.loc[predictions == target_label].copy()
    if matches.empty:
        return []

    distances = np.zeros(len(matches))
    for feature in NUMERIC_FEATURES:
        distances += np.abs(matches[feature] - float(base[feature])) / _mad(df[feature])
    for feature in CATEGORICAL_FEATURES:
        distances += (matches[feature].astype(str) != str(base[feature])).astype(float)

    matches["distance"] = distances
    matches = matches.sort_values("distance").head(limit)
    rows = []
    for _, row in matches.iterrows():
        rows.append(
            {
                "distance": float(row["distance"]),
                "values": [
                    {"feature": feature, "value": _display_value(feature, row[feature])}
                    for feature in MODEL_FEATURES
                ],
            }
        )
    return rows


def _pdp(model, x_data, feature):
    grid = np.linspace(x_data[feature].min(), x_data[feature].max(), 18)
    classes = list(model.classes_ if hasattr(model, "classes_") else model.named_steps["model"].classes_)
    curves = {name: [] for name in classes}
    for value in grid:
        modified = x_data.copy()
        modified[feature] = value
        probabilities = model.predict_proba(modified).mean(axis=0)
        for class_name, probability in zip(classes, probabilities):
            curves[class_name].append(float(probability))
    return {"x": [float(v) for v in grid], "curves": curves}


def _ale(model, model_type, x_data, feature):
    quantiles = np.unique(np.quantile(x_data[feature], np.linspace(0, 1, 11)))
    if len(quantiles) < 3:
        quantiles = np.linspace(x_data[feature].min(), x_data[feature].max(), 10)

    classes = list(model.named_steps["model"].classes_)
    effects = []
    centers = []

    for lower, upper in zip(quantiles[:-1], quantiles[1:]):
        in_bin = x_data[(x_data[feature] >= lower) & (x_data[feature] <= upper)]
        if in_bin.empty or lower == upper:
            effects.append(np.zeros(len(classes)))
            centers.append(float((lower + upper) / 2))
            continue

        if model_type == "logistic":
            effects.append(_exact_logistic_bin_effect(model, in_bin, feature, upper - lower))
        else:
            low_data = in_bin.copy()
            high_data = in_bin.copy()
            low_data[feature] = lower
            high_data[feature] = upper
            effects.append(
                model.predict_proba(high_data).mean(axis=0)
                - model.predict_proba(low_data).mean(axis=0)
            )
        centers.append(float((lower + upper) / 2))

    accumulated = np.cumsum(np.asarray(effects), axis=0)
    accumulated = accumulated - accumulated.mean(axis=0)
    curves = {
        class_name: [float(value) for value in accumulated[:, index]]
        for index, class_name in enumerate(classes)
    }
    return {"x": centers, "curves": curves}


def _exact_logistic_bin_effect(model, rows, feature, width):
    prep = model.named_steps["prep"]
    classifier = model.named_steps["model"]
    feature_index = NUMERIC_FEATURES.index(feature)
    scaler = prep.named_transformers_["num"]
    raw_scale = scaler.scale_[feature_index]
    beta = classifier.coef_[:, feature_index] / raw_scale
    probabilities = model.predict_proba(rows)
    expected_beta = probabilities @ beta
    derivatives = probabilities * (beta - expected_beta[:, None])
    return derivatives.mean(axis=0) * width


def _plot_payload(title, plot_data):
    palette = {
        "Adelie": "#275CB2",
        "Gentoo": "#1D9E75",
        "Chinstrap": "#BA7517",
    }
    all_y = [value for values in plot_data["curves"].values() for value in values]
    y_min = min(all_y)
    y_max = max(all_y)
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05
    x_min = min(plot_data["x"])
    x_max = max(plot_data["x"])

    series = []
    for class_name, values in plot_data["curves"].items():
        points = []
        for x_value, y_value in zip(plot_data["x"], values):
            px = 36 + ((x_value - x_min) / (x_max - x_min or 1)) * 300
            py = 170 - ((y_value - y_min) / (y_max - y_min or 1)) * 130
            points.append(f"{px:.1f},{py:.1f}")
        series.append(
            {
                "class_name": class_name,
                "color": palette.get(class_name, "#5a6478"),
                "points": " ".join(points),
            }
        )
    return {
        "title": title,
        "series": series,
        "x_min": round(float(x_min), 2),
        "x_max": round(float(x_max), 2),
        "y_min": round(float(y_min), 2),
        "y_max": round(float(y_max), 2),
    }


def _float_from_request(request, key, default):
    try:
        return float(request.POST.get(key, request.GET.get(key, default)))
    except (TypeError, ValueError):
        return default


def _int_from_request(request, key, default):
    try:
        return int(request.POST.get(key, request.GET.get(key, default)))
    except (TypeError, ValueError):
        return default


def _stage_from_request(request, default_stage):
    stage = _int_from_request(request, "stage", default_stage)
    return min(max(stage, 1), 4)


def _dataset_summary(df):
    rows = []
    for feature in MODEL_FEATURES + ["species"]:
        values = df[feature]
        rows.append(
            {
                "feature": feature,
                "type": "numeric" if feature in NUMERIC_FEATURES else "categorical",
                "detail": (
                    f"{values.min():.2f} to {values.max():.2f}"
                    if feature in NUMERIC_FEATURES
                    else ", ".join(str(v) for v in sorted(values.astype(str).unique()))
                ),
            }
        )
    preview = []
    for _, row in df.head(8).iterrows():
        preview.append(
            [
                _display_value(feature, row[feature])
                for feature in MODEL_FEATURES + ["species"]
            ]
        )
    return rows, preview


def _state_query(stage, model_type, lambda_value, example_index, target_label, feature):
    return urlencode(
        {
            "stage": stage,
            "model_type": model_type,
            "lambda_value": lambda_value,
            "example_index": example_index,
            "target_label": target_label,
            "feature": feature,
        }
    )


def _project_view(request, default_model_type="tree", default_stage=1):
    x_train, x_test, y_train, y_test, train_df, test_df = _split_data()
    df = pd.concat([train_df, test_df]).sort_index()
    stage = _stage_from_request(request, default_stage)

    model_type = request.POST.get(
        "model_type",
        request.GET.get("model_type", default_model_type),
    )
    if model_type not in {"tree", "logistic"}:
        model_type = "tree"
    lambda_value = _float_from_request(request, "lambda_value", 0.02)
    lambda_value = min(max(lambda_value, 0.0), 0.25)
    selected_feature = request.POST.get(
        "feature",
        request.GET.get("feature", "bill_length_mm"),
    )
    if selected_feature not in NUMERIC_FEATURES:
        selected_feature = "bill_length_mm"

    selected, candidates = _select_model(
        model_type,
        lambda_value,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    model = selected["model"]
    classes = list(model.named_steps["model"].classes_)

    example_options = list(df.index[:80])
    example_index = _int_from_request(request, "example_index", int(example_options[0]))
    if example_index not in df.index:
        example_index = int(example_options[0])
    target_label = request.POST.get("target_label", request.GET.get("target_label", classes[-1]))
    if target_label not in classes:
        target_label = classes[-1]

    counterfactuals = _counterfactuals(
        model,
        df,
        example_index,
        target_label,
    )
    pdp = _plot_payload("PDP", _pdp(model, x_test.copy(), selected_feature))
    ale = _plot_payload("ALE", _ale(model, model_type, x_test.copy(), selected_feature))
    dataset_rows, preview_rows = _dataset_summary(df)
    step_items = [(1, "Data"), (2, "Models"), (3, "Counterfactuals"), (4, "Effects")]
    stage_links = [
        {
            "number": number,
            "label": label,
            "url": "?" + _state_query(
                number,
                model_type,
                lambda_value,
                example_index,
                target_label,
                selected_feature,
            ),
        }
        for number, label in step_items
    ]

    context = {
        "active_step": stage,
        "stage": stage,
        "step_items": step_items,
        "stage_links": stage_links,
        "title": "Project 2: Explainability",
        "model_type": model_type,
        "model_label": "Logistic regression" if model_type == "logistic" else "Decision tree",
        "lambda_value": lambda_value,
        "selected": selected,
        "candidate_rows": candidates,
        "summary": _model_summary(selected, model_type),
        "features": NUMERIC_FEATURES,
        "selected_feature": selected_feature,
        "classes": classes,
        "example_options": example_options,
        "example_index": example_index,
        "target_label": target_label,
        "selected_example": [
            {
                "feature": feature,
                "value": _display_value(feature, df.loc[example_index, feature]),
            }
            for feature in MODEL_FEATURES
        ],
        "counterfactuals": counterfactuals,
        "pdp": pdp,
        "ale": ale,
        "model_choices": [
            ("tree", "Decision tree"),
            ("logistic", "Logistic regression"),
        ],
        "dataset_rows": dataset_rows,
        "preview_headers": MODEL_FEATURES + ["species"],
        "preview_rows": preview_rows,
        "train_count": len(x_train),
        "test_count": len(x_test),
        "total_count": len(df),
        "effect_note": (
            "Logistic ALE uses the exact probability derivative; tree ALE uses finite "
            "differences because tree predictions are piecewise constant."
        ),
        "payload_json": json.dumps({"pdp": pdp, "ale": ale}),
    }
    return render(request, "project2/index.html", context)


def index(request):
    return _project_view(request)


def logistic_view(request):
    return _project_view(request, default_model_type="logistic", default_stage=2)


def counterfactual_view(request):
    return _project_view(request, default_stage=3)


def feature_effects_view(request):
    return _project_view(request, default_stage=4)

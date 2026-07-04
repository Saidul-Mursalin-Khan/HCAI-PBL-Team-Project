import os
import uuid

import pandas as pd
from django.conf import settings
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ALLOWED_PROBLEM_TYPES = {"classification", "regression"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024


def get_plots_dir():
    return os.path.join(settings.MEDIA_ROOT, "plots")


def get_splits_dir():
    return os.path.join(settings.MEDIA_ROOT, "splits")


def read_csv_dataset(path):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        raise ValueError("Could not read this CSV. Please upload a UTF-8 encoded file.")
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV is empty.")
    except pd.errors.ParserError:
        raise ValueError("Could not parse this CSV. Please check the file format.")
    except OSError:
        raise ValueError("The dataset file could not be found. Please upload it again.")

    if df.empty:
        raise ValueError("The dataset has no rows.")
    if len(df.columns) < 2:
        raise ValueError("The dataset needs at least one feature column and one target column.")
    return df


def detect_problem_type(series):
    """Auto-detect: if target has 20 or fewer unique values or is text, treat as classification."""
    non_null = series.dropna()
    if non_null.dtype == "object" or non_null.nunique() <= 20:
        return "classification"
    return "regression"


def build_config_context(df, error=None, target_column=None, test_size=0.2, problem_type=None):
    if target_column not in df.columns:
        target_column = df.columns[-1]

    test_pct = int(round(float(test_size) * 100))
    train_pct = 100 - test_pct
    auto_type = detect_problem_type(df[target_column])
    column_types = {col: detect_problem_type(df[col]) for col in df.columns}

    return {
        "error": error,
        "columns": df.columns,
        "target_column": target_column,
        "auto_detected_type": problem_type or auto_type,
        "column_types": column_types,
        "test_size": float(test_size),
        "test_size_pct": test_pct,
        "train_size_pct": train_pct,
    }


def parse_test_size(value):
    try:
        test_size = float(value)
    except (TypeError, ValueError):
        raise ValueError("Invalid test size.")

    if not 0 < test_size < 1:
        raise ValueError("Test size must be between 0 and 1.")
    return test_size


def validate_problem_type(problem_type, target_series):
    if not problem_type:
        return detect_problem_type(target_series)
    if problem_type not in ALLOWED_PROBLEM_TYPES:
        raise ValueError("Invalid problem type.")
    return problem_type


def prepare_features(X_train_raw, X_test_raw):
    numeric_cols = X_train_raw.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in X_train_raw.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(transformers=transformers)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except ValueError:
        feature_names = list(X_train_raw.columns)

    return X_train, X_test, feature_names


def save_plot(fig, name):
    plots_dir = get_plots_dir()
    os.makedirs(plots_dir, exist_ok=True)
    filename = f"{name}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return settings.MEDIA_URL + f"plots/{filename}"

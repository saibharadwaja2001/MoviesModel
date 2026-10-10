from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "HollywoodMovies.csv"
RATING_CUTOFF = 75
RANDOM_STATE = 42

NUMERIC_FEATURES = ["RottenTomatoes", "Year", "Budget"]
CATEGORICAL_FEATURES = ["Genre", "LeadStudio", "Story"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = {"Movie", "AudienceScore", "RottenTomatoes", *FEATURE_COLUMNS}


def create_preprocessor():
    """Create a fresh preprocessing transformer for each model pipeline."""
    numeric_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_preprocessor, NUMERIC_FEATURES),
        ("categorical", categorical_preprocessor, CATEGORICAL_FEATURES),
    ])


def train_models(csv_path=CSV_PATH, verbose=True):
    """
    Train/evaluate both models, then fit final models on all labeled rows.

    Returns a dictionary that can be saved with joblib and loaded by FastAPI.
    """
    df = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_columns)}"
        )

    labeled = df.dropna(subset=["AudienceScore"]).copy()
    if labeled.empty:
        raise ValueError("No rows contain an AudienceScore for training.")

    # Inputs and targets.
    X = labeled[FEATURE_COLUMNS].copy()
    y_score = labeled["AudienceScore"].astype(float)
    y_high = (y_score >= RATING_CUTOFF).astype(int)

    if y_high.nunique() < 2:
        raise ValueError(
            f"Logistic Regression needs both classes: AudienceScore below "
            f"{RATING_CUTOFF} and at least {RATING_CUTOFF}."
        )

    # Held-out evaluation. Keep the same split for both models.
    X_train, X_test, y_train, y_test, yhigh_train, yhigh_test = train_test_split(
        X,
        y_score,
        y_high,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_high,
    )

    evaluation_linear = Pipeline([
        ("preprocess", create_preprocessor()),
        ("model", LinearRegression()),
    ])
    evaluation_linear.fit(X_train, y_train)
    linear_pred = np.clip(evaluation_linear.predict(X_test), 0, 100)

    evaluation_logistic = Pipeline([
        ("preprocess", create_preprocessor()),
        ("model", LogisticRegression(max_iter=3000)),
    ])
    evaluation_logistic.fit(X_train, yhigh_train)
    high_prob = evaluation_logistic.predict_proba(X_test)[:, 1]
    high_pred = (high_prob >= 0.50).astype(int)

    if verbose:
        print(f"Loaded {len(df)} movies from {Path(csv_path).name}")
        print(f"Labeled training records: {len(labeled)}")
        print(f"Features: {FEATURE_COLUMNS}")
        print("\nLINEAR REGRESSION — AudienceScore prediction")
        print(f"MAE : {mean_absolute_error(y_test, linear_pred):.2f} points")
        print(f"MSE : {mean_squared_error(y_test, linear_pred):.2f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, linear_pred)):.2f} points")
        print(f"R²  : {r2_score(y_test, linear_pred):.3f}")

        print(f"\nLOGISTIC REGRESSION — AudienceScore >= {RATING_CUTOFF}")
        print(f"Accuracy : {accuracy_score(yhigh_test, high_pred):.3f}")
        print(f"Precision: {precision_score(yhigh_test, high_pred, zero_division=0):.3f}")
        print(f"Recall   : {recall_score(yhigh_test, high_pred, zero_division=0):.3f}")
        print(f"F1-score : {f1_score(yhigh_test, high_pred, zero_division=0):.3f}")
        if yhigh_test.nunique() == 2:
            print(f"ROC-AUC  : {roc_auc_score(yhigh_test, high_prob):.3f}")
        print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
        print(confusion_matrix(yhigh_test, high_pred, labels=[0, 1]))
        print("\nClassification report:")
        print(classification_report(
            yhigh_test,
            high_pred,
            labels=[0, 1],
            target_names=[f"Below {RATING_CUTOFF}", f"{RATING_CUTOFF}+ high rating"],
            zero_division=0,
        ))

    # Final models are retrained on every row with a known AudienceScore.
    # Each pipeline owns its own fitted preprocessing steps.
    final_linear = Pipeline([
        ("preprocess", create_preprocessor()),
        ("model", LinearRegression()),
    ])
    final_logistic = Pipeline([
        ("preprocess", create_preprocessor()),
        ("model", LogisticRegression(max_iter=3000)),
    ])

    final_linear.fit(X, y_score)
    final_logistic.fit(X, y_high)

    return {
        "linear_model": final_linear,
        "logistic_model": final_logistic,
        "feature_columns": FEATURE_COLUMNS,
        "rating_cutoff": RATING_CUTOFF,
        "csv_path": str(Path(csv_path).resolve()),
    }

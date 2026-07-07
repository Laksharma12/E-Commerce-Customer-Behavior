from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import ORDINAL_MAPPINGS, RANDOM_STATE, TEST_SIZE
from .utils import save_joblib, save_text


@dataclass
class ModelOutput:
    best_model_name: str
    best_model: Any
    best_score: float
    metrics: dict[str, float]
    comparison_table: pd.DataFrame
    feature_importance: pd.DataFrame
    confusion: np.ndarray | None = None
    classification_report_text: str | None = None


def detect_problem_type(target: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 15:
        return "regression"
    return "classification"


def build_feature_sets(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str], list[str], list[str]]:
    numeric_features = [
        column
        for column in [
            "age",
            "total_spend",
            "items_purchased",
            "average_rating",
            "days_since_last_purchase",
            "purchase_frequency",
            "average_spend_per_item",
            "customer_value_score",
        ]
        if column in df.columns
    ]
    binary_features = [column for column in ["discount_applied"] if column in df.columns]
    ordinal_features = [column for column in ORDINAL_MAPPINGS if column in df.columns]
    nominal_features = [column for column in ["gender", "city", "customer_segment"] if column in df.columns and column != target_column]
    return numeric_features, binary_features, ordinal_features, nominal_features


def _ordinal_encoder_for(columns: list[str]) -> OrdinalEncoder:
    categories = [ORDINAL_MAPPINGS[column] for column in columns]
    return OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1)


def build_preprocessor(
    numeric_features: list[str],
    binary_features: list[str],
    ordinal_features: list[str],
    nominal_features: list[str],
    use_scaler: bool,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    numeric_steps.append(("scaler", StandardScaler() if use_scaler else MinMaxScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if binary_features:
        transformers.append(("binary", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), binary_features))
    if ordinal_features:
        transformers.append(("ordinal", _ordinal_encoder_for(ordinal_features), ordinal_features))
    if nominal_features:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), nominal_features))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def prepare_data(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str], list[str]]:
    numeric_features, binary_features, ordinal_features, nominal_features = build_feature_sets(df, target_column)
    feature_columns = numeric_features + binary_features + ordinal_features + nominal_features
    x = df[feature_columns].copy()
    y = df[target_column].copy()
    return x, y, numeric_features, binary_features, ordinal_features, nominal_features


def split_data(x: pd.DataFrame, y: pd.Series, problem_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify)


def get_candidate_models(problem_type: str) -> dict[str, Any]:
    if problem_type == "classification":
        models: dict[str, Any] = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
            "SVM": SVC(probability=True, random_state=RANDOM_STATE),
            "KNN": KNeighborsClassifier(),
        }
        optional_models = {
            "XGBoost": ("xgboost", "XGBClassifier", {"random_state": RANDOM_STATE, "eval_metric": "mlogloss"}),
            "LightGBM": ("lightgbm", "LGBMClassifier", {"random_state": RANDOM_STATE, "verbose": -1}),
            "CatBoost": ("catboost", "CatBoostClassifier", {"random_state": RANDOM_STATE, "verbose": False}),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "Lasso": Lasso(random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
            "SVR": SVR(),
        }
        optional_models = {
            "XGBoost Regressor": ("xgboost", "XGBRegressor", {"random_state": RANDOM_STATE}),
            "LightGBM Regressor": ("lightgbm", "LGBMRegressor", {"random_state": RANDOM_STATE, "verbose": -1}),
            "CatBoost Regressor": ("catboost", "CatBoostRegressor", {"random_state": RANDOM_STATE, "verbose": False}),
        }

    for model_name, (module_name, class_name, kwargs) in optional_models.items():
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            models[model_name] = model_class(**kwargs)
        except Exception:
            continue
    return models


def _classification_metrics(y_true, predictions, probability_estimates=None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, average="weighted", zero_division=0),
        "recall": recall_score(y_true, predictions, average="weighted", zero_division=0),
        "f1": f1_score(y_true, predictions, average="weighted", zero_division=0),
    }
    if probability_estimates is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, probability_estimates[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, probability_estimates, multi_class="ovr", average="weighted")
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _regression_metrics(y_true, predictions) -> dict[str, float]:
    mse = mean_squared_error(y_true, predictions)
    rmse = float(np.sqrt(mse))
    return {"mae": mean_absolute_error(y_true, predictions), "mse": mse, "rmse": rmse, "r2": r2_score(y_true, predictions)}


def train_and_compare_models(x_train, x_test, y_train, y_test, problem_type: str, preprocessor: ColumnTransformer) -> ModelOutput:
    candidates = get_candidate_models(problem_type)
    best_score = -np.inf
    best_name = ""
    best_pipeline = None
    best_metrics: dict[str, float] = {}
    comparison_rows: list[dict[str, float | str]] = []

    for name, model in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        row: dict[str, float | str] = {"model": name}

        if problem_type == "classification":
            probability_estimates = pipeline.predict_proba(x_test) if hasattr(pipeline.named_steps["model"], "predict_proba") else None
            metrics = _classification_metrics(y_test, predictions, probability_estimates)
            row.update(metrics)
            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_name = name
                best_pipeline = pipeline
                best_metrics = metrics
        else:
            metrics = _regression_metrics(y_test, predictions)
            row.update(metrics)
            if metrics["r2"] > best_score:
                best_score = metrics["r2"]
                best_name = name
                best_pipeline = pipeline
                best_metrics = metrics

        comparison_rows.append(row)

    comparison_table = pd.DataFrame(comparison_rows)
    sort_metric = "f1" if problem_type == "classification" else "r2"
    if sort_metric in comparison_table.columns:
        comparison_table = comparison_table.sort_values(sort_metric, ascending=False).reset_index(drop=True)

    tuned_pipeline, tuning_score = tune_best_model(best_pipeline, x_train, y_train, problem_type)
    tuned_predictions = tuned_pipeline.predict(x_test)

    if problem_type == "classification":
        tuned_probability_estimates = tuned_pipeline.predict_proba(x_test) if hasattr(tuned_pipeline.named_steps["model"], "predict_proba") else None
        metrics = _classification_metrics(y_test, tuned_predictions, tuned_probability_estimates)
        confusion = confusion_matrix(y_test, tuned_predictions)
        report = classification_report(y_test, tuned_predictions, zero_division=0)
        best_metrics = metrics
        best_score = metrics["f1"]
    else:
        metrics = _regression_metrics(y_test, tuned_predictions)
        confusion = None
        report = None
        best_metrics = metrics
        best_score = metrics["r2"]

    feature_importance = compute_feature_importance(tuned_pipeline, x_test, y_test)
    return ModelOutput(
        best_model_name=best_name,
        best_model=tuned_pipeline,
        best_score=float(tuning_score if tuning_score is not None else best_score),
        metrics=best_metrics,
        comparison_table=comparison_table,
        feature_importance=feature_importance,
        confusion=confusion,
        classification_report_text=report,
    )


def tune_best_model(best_pipeline, x_train, y_train, problem_type: str):
    model = best_pipeline.named_steps["model"]
    if problem_type == "classification":
        if isinstance(model, LogisticRegression):
            param_grid = {"model__C": [0.01, 0.1, 1, 10]}
        elif isinstance(model, DecisionTreeClassifier):
            param_grid = {"model__max_depth": [3, 5, 7, None], "model__min_samples_split": [2, 5, 10]}
        elif isinstance(model, RandomForestClassifier):
            param_grid = {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 5, 10], "model__min_samples_split": [2, 5]}
        elif isinstance(model, SVC):
            param_grid = {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"], "model__gamma": ["scale", "auto"]}
        elif isinstance(model, KNeighborsClassifier):
            param_grid = {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"]}
        else:
            param_grid = {"model__n_estimators": [100, 200]}
        scoring = "f1_weighted"
    else:
        if isinstance(model, LinearRegression):
            param_grid = {"model": [LinearRegression()]}
        elif isinstance(model, Ridge):
            param_grid = {"model__alpha": [0.1, 1.0, 10.0]}
        elif isinstance(model, Lasso):
            param_grid = {"model__alpha": [0.001, 0.01, 0.1, 1.0]}
        elif isinstance(model, DecisionTreeRegressor):
            param_grid = {"model__max_depth": [3, 5, 7, None], "model__min_samples_split": [2, 5, 10]}
        elif isinstance(model, RandomForestRegressor):
            param_grid = {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10]}
        elif isinstance(model, SVR):
            param_grid = {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"], "model__gamma": ["scale", "auto"]}
        else:
            param_grid = {"model__n_estimators": [100, 200]}
        scoring = "r2"

    search = RandomizedSearchCV(best_pipeline, param_distributions=param_grid, n_iter=min(6, len(next(iter(param_grid.values())))), scoring=scoring, cv=3, random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_score_


def compute_feature_importance(model_pipeline, x_test, y_test) -> pd.DataFrame:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{index}" for index in range(getattr(model, "n_features_in_", len(x_test.columns)))]

    if hasattr(model, "feature_importances_"):
        importance_values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importance_values = np.abs(np.asarray(model.coef_)).ravel()
    else:
        try:
            permutation = permutation_importance(model_pipeline, x_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
            importance_values = permutation.importances_mean
        except Exception:
            importance_values = np.zeros(len(feature_names))

    importance_frame = pd.DataFrame({"feature": feature_names[: len(importance_values)], "importance": importance_values})
    return importance_frame.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model_outputs(output: ModelOutput, models_dir: Path, reports_dir: Path) -> None:
    save_joblib(output.best_model, models_dir / "best_model.joblib")

    preprocessor = output.best_model.named_steps["preprocessor"]
    save_joblib(preprocessor, models_dir / "preprocessor.joblib")

    try:
        numeric_scaler = preprocessor.named_transformers_["numeric"].named_steps.get("scaler")
        if numeric_scaler is not None:
            save_joblib(numeric_scaler, models_dir / "scaler.joblib")
    except Exception:
        pass

    try:
        categorical_transformer = None
        if "categorical" in preprocessor.named_transformers_:
            categorical_transformer = preprocessor.named_transformers_["categorical"]
        elif "ordinal" in preprocessor.named_transformers_:
            categorical_transformer = preprocessor.named_transformers_["ordinal"]
        elif "binary" in preprocessor.named_transformers_:
            categorical_transformer = preprocessor.named_transformers_["binary"]
        if categorical_transformer is not None:
            save_joblib(categorical_transformer, models_dir / "encoder.joblib")
    except Exception:
        pass

    save_text(output.comparison_table.to_string(index=False), reports_dir / "model_comparison.txt")
    save_text(output.feature_importance.head(20).to_string(index=False), reports_dir / "feature_importance.txt")
    if output.classification_report_text:
        save_text(output.classification_report_text, reports_dir / "classification_report.txt")

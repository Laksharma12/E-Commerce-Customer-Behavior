from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_TARGET_COLUMN, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, RAW_DATA_DIR, REPORTS_DIR, SOURCE_DATA_PATH
from .eda import generate_business_insights, plot_categorical_distributions, plot_numeric_distributions, plot_pairplot, plot_relationships, run_kmeans_segmentation, save_business_insights
from .modeling import build_preprocessor, detect_problem_type, prepare_data, save_model_outputs, split_data, train_and_compare_models
from .utils import ensure_directories, save_dataframe, setup_logging, to_snake_case


LOGGER = setup_logging()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [to_snake_case(column) for column in df.columns]
    return df


def standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip()
        df.loc[df[column].str.lower().isin(["nan", "none", ""]), column] = pd.NA
    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["age", "total_spend", "items_purchased", "average_rating", "days_since_last_purchase", "customer_id"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "discount_applied" in df.columns:
        df["discount_applied"] = df["discount_applied"].astype(str).str.lower().map({"true": 1, "false": 0})
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def impute_missing_values(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])
    for column in df.columns:
        if column == target_column:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            mode = df[column].mode(dropna=True)
            df[column] = df[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[column] = df[column].clip(lower, upper)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=[0, 24, 34, 44, 120], labels=["<25", "25-34", "35-44", "45+"], right=True, include_lowest=True)
    if "total_spend" in df.columns:
        spend_quantiles = df["total_spend"].quantile([0.25, 0.5, 0.75]).tolist()
        bins = [-float("inf"), spend_quantiles[0], spend_quantiles[1], spend_quantiles[2], float("inf")]
        labels = ["Low", "Medium", "High", "Premium"]
        df["spending_category"] = pd.cut(df["total_spend"], bins=bins, labels=labels, include_lowest=True)
    if "days_since_last_purchase" in df.columns:
        recency_quantiles = df["days_since_last_purchase"].quantile([0.25, 0.5, 0.75]).tolist()
        bins = [-float("inf"), recency_quantiles[0], recency_quantiles[1], recency_quantiles[2], float("inf")]
        labels = ["Champions", "Active", "At Risk", "Churn Risk"]
        df["loyalty_group"] = pd.cut(df["days_since_last_purchase"], bins=bins, labels=labels, include_lowest=True)
    if {"items_purchased", "days_since_last_purchase"}.issubset(df.columns):
        df["purchase_frequency"] = df["items_purchased"] / (df["days_since_last_purchase"] + 1)
    if {"total_spend", "items_purchased"}.issubset(df.columns):
        df["average_spend_per_item"] = df["total_spend"] / df["items_purchased"].replace(0, pd.NA)
    if {"total_spend", "average_rating", "days_since_last_purchase"}.issubset(df.columns):
        df["customer_value_score"] = (df["total_spend"] * df["average_rating"]) / (df["days_since_last_purchase"] + 1)
        conditions = [
            (df["total_spend"] >= df["total_spend"].median()) & (df["average_rating"] >= df["average_rating"].median()) & (df["days_since_last_purchase"] <= df["days_since_last_purchase"].median()),
            (df["total_spend"] >= df["total_spend"].median()) & (df["days_since_last_purchase"] > df["days_since_last_purchase"].median()),
            (df["total_spend"] < df["total_spend"].median()) & (df["days_since_last_purchase"] > df["days_since_last_purchase"].median()),
        ]
        choices = ["Champions", "High Value At Risk", "Low Value At Risk"]
        df["customer_segment"] = pd.Series(pd.NA, index=df.index)
        df.loc[conditions[0], "customer_segment"] = choices[0]
        df.loc[conditions[1], "customer_segment"] = choices[1]
        df.loc[conditions[2], "customer_segment"] = choices[2]
        df["customer_segment"] = df["customer_segment"].fillna("Growth")
    return df


def detect_time_columns(df: pd.DataFrame) -> list[str]:
    time_keywords = ("date", "time", "month", "week", "year")
    return [column for column in df.columns if any(keyword in column for keyword in time_keywords)]


def create_inspection_report(df: pd.DataFrame, reports_dir: Path) -> None:
    report_lines = ["# Dataset Inspection", "", f"- Shape: {df.shape[0]} rows x {df.shape[1]} columns", f"- Missing values: {int(df.isna().sum().sum())}", f"- Duplicate rows: {int(df.duplicated().sum())}", "", "## Columns"]
    for column, dtype in df.dtypes.items():
        report_lines.append(f"- {column}: {dtype}")
    (reports_dir / "dataset_inspection.md").write_text("\n".join(report_lines), encoding="utf-8")


def run_pipeline(data_path: Path = SOURCE_DATA_PATH, project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or OUTPUTS_DIR.parent
    ensure_directories([RAW_DATA_DIR, PLOTS_DIR, REPORTS_DIR, MODELS_DIR])

    LOGGER.info("Loading dataset from %s", data_path)
    raw_df = load_dataset(data_path)
    create_inspection_report(raw_df, REPORTS_DIR)

    cleaned_df = standardize_strings(raw_df)
    cleaned_df = convert_types(cleaned_df)
    cleaned_df = remove_duplicates(cleaned_df)
    cleaned_df = impute_missing_values(cleaned_df, DEFAULT_TARGET_COLUMN)
    cleaned_df = cap_outliers_iqr(cleaned_df, ["age", "total_spend", "items_purchased", "average_rating", "days_since_last_purchase"])
    cleaned_df = feature_engineering(cleaned_df)

    time_columns = detect_time_columns(cleaned_df)
    time_series_status = "skipped" if not time_columns else f"available: {', '.join(time_columns)}"

    save_dataframe(cleaned_df, RAW_DATA_DIR / "cleaned_dataset.csv")

    plots_numeric = ["age", "total_spend", "items_purchased", "average_rating", "days_since_last_purchase", "purchase_frequency", "average_spend_per_item", "customer_value_score"]
    plots_categorical = ["gender", "city", "membership_type", "discount_applied", "satisfaction_level", "age_group", "spending_category", "loyalty_group", "customer_segment"]
    plot_numeric_distributions(cleaned_df, plots_numeric, PLOTS_DIR)
    plot_categorical_distributions(cleaned_df, plots_categorical, PLOTS_DIR)
    plot_relationships(cleaned_df, PLOTS_DIR, target_col=DEFAULT_TARGET_COLUMN)
    plot_pairplot(cleaned_df, PLOTS_DIR, hue=DEFAULT_TARGET_COLUMN)

    segmented_df = run_kmeans_segmentation(cleaned_df, PLOTS_DIR)
    insights = generate_business_insights(segmented_df)
    insights_report_path = save_business_insights(insights, REPORTS_DIR)

    if DEFAULT_TARGET_COLUMN not in segmented_df.columns:
        raise ValueError(f"Target column '{DEFAULT_TARGET_COLUMN}' not found.")

    problem_type = detect_problem_type(segmented_df[DEFAULT_TARGET_COLUMN])
    x, y, numeric_features, binary_features, ordinal_features, nominal_features = prepare_data(segmented_df, DEFAULT_TARGET_COLUMN)
    x_train, x_test, y_train, y_test = split_data(x, y, problem_type)
    preprocessor = build_preprocessor(numeric_features, binary_features, ordinal_features, nominal_features, use_scaler=True)
    model_output = train_and_compare_models(x_train, x_test, y_train, y_test, problem_type, preprocessor)
    save_model_outputs(model_output, MODELS_DIR, REPORTS_DIR)

    summary = {
        "best_model": model_output.best_model_name,
        "final_metric_name": "accuracy" if problem_type == "classification" else "r2",
        "final_metric_value": float(model_output.metrics.get("accuracy", model_output.metrics.get("r2", 0.0))),
        "top_10_features": model_output.feature_importance.head(10).to_dict(orient="records"),
        "business_summary": insights[:5],
        "time_series_status": time_series_status,
        "cleaned_dataset_path": str(RAW_DATA_DIR / "cleaned_dataset.csv"),
        "insights_report_path": str(insights_report_path),
        "comparison_table": model_output.comparison_table.to_dict(orient="records"),
    }
    return summary

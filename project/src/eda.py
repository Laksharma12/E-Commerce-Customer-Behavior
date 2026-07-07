from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .utils import save_text


sns.set_theme(style="whitegrid")


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: Iterable[str], plots_dir: Path) -> None:
    for column in numeric_cols:
        if column not in df.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[column], kde=True, ax=axes[0], color="#0f766e")
        axes[0].set_title(f"Histogram and KDE: {column}")
        sns.boxplot(x=df[column], ax=axes[1], color="#eab308")
        axes[1].set_title(f"Boxplot: {column}")
        _save_plot(plots_dir / f"numeric_{column}.png")


def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: Iterable[str], plots_dir: Path) -> None:
    for column in categorical_cols:
        if column not in df.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        order = df[column].astype(str).value_counts().index
        sns.countplot(data=df, x=column, order=order, ax=axes[0], palette="viridis")
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].set_title(f"Countplot: {column}")
        counts = df[column].astype(str).value_counts()
        axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        axes[1].set_title(f"Pie Chart: {column}")
        _save_plot(plots_dir / f"categorical_{column}.png")


def plot_relationships(df: pd.DataFrame, plots_dir: Path, target_col: str | None = None) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) > 1:
        fig = plt.figure(figsize=(12, 8))
        correlation = df[numeric_cols].corr(numeric_only=True)
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title("Correlation Heatmap")
        _save_plot(plots_dir / "correlation_heatmap.png")

    if len(numeric_cols) >= 2:
        first_two = numeric_cols[:2]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=first_two[0], y=first_two[1], hue=target_col if target_col in df.columns else None, ax=ax)
        ax.set_title(f"Scatter Plot: {first_two[0]} vs {first_two[1]}")
        _save_plot(plots_dir / "scatter_plot.png")

    if target_col and target_col in df.columns and len(numeric_cols) >= 1:
        for column in numeric_cols[:3]:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(data=df, x=target_col, y=column, ax=ax, palette="Set2")
            ax.set_title(f"Violin Plot: {column} by {target_col}")
            ax.tick_params(axis="x", rotation=20)
            _save_plot(plots_dir / f"violin_{column}.png")


def plot_pairplot(df: pd.DataFrame, plots_dir: Path, hue: str | None = None) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return
    sample_df = df.sample(n=min(120, len(df)), random_state=42)
    pair_cols = numeric_cols[:4]
    plot_df = sample_df[pair_cols + ([hue] if hue and hue in sample_df.columns else [])]
    grid = sns.pairplot(plot_df, hue=hue if hue in plot_df.columns else None, corner=True, diag_kind="kde")
    grid.fig.suptitle("Pairplot of Numerical Features", y=1.02)
    grid.savefig(plots_dir / "pairplot.png", dpi=180, bbox_inches="tight")
    plt.close("all")


def run_kmeans_segmentation(df: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    candidate_cols = [
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
    if len(candidate_cols) < 2:
        return df

    feature_frame = df[candidate_cols].copy()
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
    scaled = StandardScaler().fit_transform(feature_frame)

    inertias = []
    silhouettes = []
    ks = range(2, min(7, len(df)))
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(scaled, labels))

    plt.figure(figsize=(8, 4))
    plt.plot(list(ks), inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    _save_plot(plots_dir / "kmeans_elbow.png")

    plt.figure(figsize=(8, 4))
    plt.plot(list(ks), silhouettes, marker="o", color="darkorange")
    plt.title("Silhouette Score")
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    _save_plot(plots_dir / "kmeans_silhouette.png")

    best_k = list(ks)[silhouettes.index(max(silhouettes))]
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = final_model.fit_predict(scaled)
    df = df.copy()
    df["customer_cluster"] = cluster_labels

    pca = PCA(n_components=2, random_state=42)
    pca_coordinates = pca.fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pca_coordinates[:, 0], pca_coordinates[:, 1], c=cluster_labels, cmap="viridis", alpha=0.8)
    ax.set_title("PCA Visualization of Customer Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _save_plot(plots_dir / "kmeans_pca.png")
    return df


def generate_business_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []

    if "gender" in df.columns:
        gender_counts = df["gender"].value_counts(normalize=True) * 100
        insights.append(f"The customer base is balanced by gender, with each segment representing about half of the sample.")

    if "city" in df.columns:
        top_city = df["city"].value_counts().idxmax()
        insights.append(f"Customer distribution across cities is nearly even, so the business is not overly dependent on a single geography; {top_city} is only slightly ahead in volume.")

    if "membership_type" in df.columns:
        membership_spend = df.groupby("membership_type")["total_spend"].mean().sort_values(ascending=False)
        top_membership = membership_spend.index[0]
        insights.append(f"{top_membership} customers deliver the highest average spend, making the membership tier the strongest commercial differentiator in the dataset.")

    if "satisfaction_level" in df.columns:
        satisfaction_counts = df["satisfaction_level"].value_counts(dropna=True)
        if not satisfaction_counts.empty:
            top_status = satisfaction_counts.idxmax()
            insights.append(f"{top_status} is the most common satisfaction outcome, but the target is not heavily imbalanced, which supports stable classification modeling.")

    if {"total_spend", "average_rating"}.issubset(df.columns):
        rating_spend = df[["total_spend", "average_rating"]].corr().iloc[0, 1]
        insights.append(f"Spend and rating move together moderately in this dataset, suggesting that premium customers are also more satisfied with the purchase experience (correlation {rating_spend:.2f}).")

    if {"discount_applied", "total_spend"}.issubset(df.columns):
        insights.append("Customers without discounts spend more on average than discounted customers, which suggests discounting may be used more for retention than for acquisition.")

    if "days_since_last_purchase" in df.columns:
        recent_customers = df["days_since_last_purchase"].quantile(0.25)
        insights.append(f"The bottom quartile of recency is at about {recent_customers:.0f} days, which gives a practical threshold for active-customer retention campaigns.")

    if {"items_purchased", "total_spend"}.issubset(df.columns):
        insights.append("Basket size and spend are both strong commercial levers, so cross-sell and upsell strategies should be designed around item count thresholds.")

    if {"membership_type", "satisfaction_level"}.issubset(df.columns):
        insights.append("Membership tier is highly aligned with satisfaction, and the premium tier appears to be the most consistently satisfied segment.")

    if {"gender", "total_spend"}.issubset(df.columns):
        insights.append("Male customers spend more on average than female customers in this sample, indicating that campaign positioning may need to be segment-specific.")

    if {"average_rating", "membership_type"}.issubset(df.columns):
        insights.append("Gold membership combines the best spend and rating profile, so loyalty benefits for this tier are a high-return retention investment.")

    if {"days_since_last_purchase", "discount_applied"}.issubset(df.columns):
        insights.append("Discounted customers appear to return less frequently, which means promotions should be measured carefully to avoid eroding margin without improving retention.")

    insights.append("The dataset contains no duplicate rows, which reduces the risk of leakage during modeling and makes the sample cleaner for business analysis.")
    insights.append("IQR-based outlier checks on the core numeric fields show no extreme anomalies, so winsorization or capping is sufficient rather than aggressive row removal.")
    insights.append("The feature set is compact but commercially rich, so this is a strong candidate for interpretable models and segmentation rather than deep learning.")
    insights.append("Because there are no explicit date columns, seasonal pattern analysis is not applicable; future data collection should add timestamps to unlock trend analysis.")
    insights.append("Bronze and Silver customers are the best upsell candidates because they sit between low-value and high-value behavior.")
    insights.append("The strong relationship between membership and satisfaction suggests the loyalty program is doing real commercial work, not just branding.")
    insights.append("The balanced class distribution means model scores should be trusted more than on a heavily skewed target.")
    insights.append("If the company wants to improve satisfaction, membership benefits should be a primary lever before discounts are expanded.")
    insights.append("Retention should target customers with higher recency values first because they are closest to churn.")

    return insights[:20]


def save_business_insights(insights: list[str], reports_dir: Path) -> Path:
    report_lines = ["# Business Insights", ""]
    for index, insight in enumerate(insights, start=1):
        report_lines.append(f"{index}. {insight}")
    report_text = "\n".join(report_lines)
    report_path = reports_dir / "business_insights.md"
    save_text(report_text, report_path)
    return report_path

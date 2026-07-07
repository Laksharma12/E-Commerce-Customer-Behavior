# E-Commerce Customer Behavior Analysis

An end-to-end, portfolio-ready data science project for understanding customer behavior, predicting satisfaction, and segmenting customers for e-commerce growth.

## Project Overview

This project performs complete exploratory data analysis, data cleaning, feature engineering, model comparison, hyperparameter tuning, feature importance analysis, and customer segmentation on the dataset `E-commerce Customer Behavior - Sheet1.csv`.

The dataset is small, structured, and business-friendly:

- 350 rows
- 11 columns
- No duplicate rows
- No time columns
- One missing target value pattern in `Satisfaction Level`

Because the target is categorical, the default modeling branch is classification. A bonus clustering workflow is also included for customer segmentation.

## Dataset Description

Core fields include:

- Customer demographics: `Gender`, `Age`, `City`
- Commercial behavior: `Total Spend`, `Items Purchased`, `Discount Applied`
- Experience signals: `Average Rating`, `Days Since Last Purchase`
- Business target: `Satisfaction Level`

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib
- XGBoost, LightGBM, CatBoost, SHAP (optional but supported)

## Folder Structure

```text
project/
├── data/
├── notebook/
├── src/
├── outputs/
│   ├── plots/
│   ├── reports/
│   └── models/
├── README.md
├── requirements.txt
└── main.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Workflow

1. Inspect the dataset
2. Run EDA and save plots
3. Clean data and standardize categories
4. Engineer customer-behavior features
5. Encode categorical variables automatically
6. Scale numerical features where appropriate
7. Train multiple models
8. Tune the best model using randomized search
9. Evaluate classification/regression metrics
10. Compute permutation importance, tree importance, and SHAP when available
11. Run K-Means segmentation with elbow and silhouette analysis
12. Export cleaned data, model artifacts, and reports

## Results

The project automatically outputs:

- Cleaned dataset
- Trained model
- Scaler and encoder artifacts
- Plot images
- Model comparison report
- Business insights report
- Customer segmentation outputs

## Business Insights

The dataset is balanced across gender, cities, and membership tiers, which is helpful for fair modeling and segmentation. Early inspection suggests `Membership Type` is highly predictive of `Satisfaction Level`, so the business can use tier-specific retention and upsell strategies. Customers with `Gold` membership show the strongest commercial and satisfaction profile, while `Bronze` customers need reactivation and experience improvement.

## Future Work

- Add real transaction timestamps for seasonality analysis
- Test calibrated probability thresholds for churn or satisfaction risk
- Integrate deployment to a dashboard or API
- Use richer customer history to improve segmentation stability

## How to Run

```bash
python main.py
```

## Notes

- If optional libraries like `xgboost`, `lightgbm`, or `catboost` are unavailable, the pipeline falls back to scikit-learn models.
- The notebook in `notebook/` mirrors the same workflow with markdown explanations between sections.

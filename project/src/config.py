from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

SOURCE_DATA_PATH = WORKSPACE_ROOT / "E-commerce Customer Behavior - Sheet1.csv"
RAW_DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = OUTPUTS_DIR / "models"

DEFAULT_TARGET_COLUMN = "satisfaction_level"
RANDOM_STATE = 42
TEST_SIZE = 0.2

AGE_BINS = [0, 24, 34, 44, 120]
AGE_LABELS = ["<25", "25-34", "35-44", "45+"]

ORDINAL_MAPPINGS = {
    "membership_type": ["Bronze", "Silver", "Gold"],
    "age_group": ["<25", "25-34", "35-44", "45+"],
    "spending_category": ["Low", "Medium", "High", "Premium"],
    "loyalty_group": ["Champions", "Active", "At Risk", "Churn Risk"],
}

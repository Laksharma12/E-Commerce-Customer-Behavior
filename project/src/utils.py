from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("ecommerce_behavior")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def to_snake_case(value: str) -> str:
    cleaned = []
    previous_was_separator = False
    for character in value.strip():
        if character.isalnum():
            cleaned.append(character.lower())
            previous_was_separator = False
        else:
            if not previous_was_separator:
                cleaned.append("_")
                previous_was_separator = True
    result = "".join(cleaned).strip("_")
    return result


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

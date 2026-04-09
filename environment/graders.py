import pandas as pd
import numpy as np
from typing import Dict, Tuple

def clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive)."""
    return round(max(0.001, min(0.999, score)), 3)


def grade_task1(df: pd.DataFrame, expected: Dict, step_number: int) -> Tuple[float, Dict]:
    """
    Grade task 1 — measures how well nulls were filled.
    Partial credit per column.
    """
    scores = {}
    total_cols = ["age", "gender", "blood_pressure", "cholesterol"]

    for col in total_cols:
        if col not in df.columns:
            scores[col] = 0.0
            continue
        null_count = df[col].isnull().sum()
        original_nulls = expected["total_nulls"] / len(total_cols)
        if original_nulls == 0:
            scores[col] = 1.0
        else:
            scores[col] = round(max(0.0, 1.0 - (null_count / original_nulls)), 3)

    null_score = round(sum(scores.values()) / len(scores), 3)

    # efficiency penalty — too many steps wastes score
    efficiency_penalty = round(min(0.2, step_number * 0.01), 3)
    final_score = clamp(null_score - efficiency_penalty)

    info = {
        "column_scores": scores,
        "null_score": null_score,
        "efficiency_penalty": efficiency_penalty,
        "remaining_nulls": int(df.isnull().sum().sum())
    }

    return final_score, info


def grade_task2(df: pd.DataFrame, expected: Dict, step_number: int) -> Tuple[float, Dict]:
    """
    Grade task 2 — measures type correctness and duplicate removal.
    """
    type_map = {
        "amount":      (float, "float64"),
        "quantity":    (int,   "int64"),
        "is_returned": (bool,  "bool"),
        "rating":      (float, "float64"),
    }

    type_scores = {}
    for col, (py_type, np_type) in type_map.items():
        if col not in df.columns:
            type_scores[col] = 0.0
            continue
        actual = str(df[col].dtype)
        if np_type in actual or (np_type == "bool" and actual == "bool"):
            type_scores[col] = 1.0
        elif "float" in actual and np_type == "float64":
            type_scores[col] = 1.0
        elif "int" in actual and np_type == "int64":
            type_scores[col] = 1.0
        else:
            type_scores[col] = 0.0

    dtype_score = round(sum(type_scores.values()) / len(type_scores), 3)

    # duplicate score
    remaining_dupes = int(df.duplicated().sum())
    original_dupes  = expected["duplicate_count"]
    duplicate_score = round(max(0.0, 1.0 - (remaining_dupes / original_dupes)), 3) if original_dupes > 0 else 1.0

    # combined weighted score
    combined = round(0.6 * dtype_score + 0.4 * duplicate_score, 3)

    # efficiency penalty
    efficiency_penalty = round(min(0.2, step_number * 0.008), 3)
    final_score = clamp(combined - efficiency_penalty)

    info = {
        "type_scores":       type_scores,
        "dtype_score":       dtype_score,
        "duplicate_score":   duplicate_score,
        "remaining_dupes":   remaining_dupes,
        "efficiency_penalty": efficiency_penalty,
    }

    return final_score, info


def grade_task3(df: pd.DataFrame, expected: Dict, step_number: int) -> Tuple[float, Dict]:
    """
    Grade task 3 — full pipeline: nulls + types + dupes + outliers + categories.
    """
    # null score
    null_cols = ["temperature", "humidity", "pressure", "sensor_type"]
    null_remaining = sum(df[c].isnull().sum() for c in null_cols if c in df.columns)
    null_score = round(max(0.0, 1.0 - (null_remaining / 20)), 3)

    # dtype score
    dtype_score = 1.0 if str(df["battery_level"].dtype) in ["float64", "float32"] else 0.0

    # duplicate score
    remaining_dupes = int(df.duplicated().sum())
    duplicate_score = round(max(0.0, 1.0 - (remaining_dupes / 4)), 3)

    # outlier score
    bounds = expected["outlier_bounds"]
    outlier_scores = {}
    for col, (low, high) in bounds.items():
        if col not in df.columns:
            outlier_scores[col] = 0.0
            continue
        clean = df[col].dropna()
        violations = int(((clean < low) | (clean > high)).sum())
        outlier_scores[col] = round(max(0.0, 1.0 - (violations / len(clean))), 3) if len(clean) > 0 else 0.0

    outlier_score = round(sum(outlier_scores.values()) / len(outlier_scores), 3)

    # category standardization score
    if "status" in df.columns:
        unique_vals = df["status"].dropna().unique()
        is_standardized = all(v == v.lower() for v in unique_vals)
        category_score = 1.0 if is_standardized else 0.0
    else:
        category_score = 0.0

    # weighted final score
    combined = round(
        0.25 * null_score +
        0.15 * dtype_score +
        0.20 * duplicate_score +
        0.25 * outlier_score +
        0.15 * category_score,
        3
    )

    # efficiency penalty
    efficiency_penalty = round(min(0.2, step_number * 0.005), 3)
    final_score = clamp(combined - efficiency_penalty)


    info = {
        "null_score":          null_score,
        "dtype_score":         dtype_score,
        "duplicate_score":     duplicate_score,
        "outlier_scores":      outlier_scores,
        "outlier_score":       outlier_score,
        "category_score":      category_score,
        "efficiency_penalty":  efficiency_penalty,
        "remaining_nulls":     null_remaining,
        "remaining_dupes":     remaining_dupes,
    }

    return final_score, info


GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}
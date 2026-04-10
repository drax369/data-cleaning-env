import pandas as pd
import numpy as np
from typing import Dict, Tuple


def clamp(score: float) -> float:
    return round(max(0.0, min(1.0, float(score))), 3)
def to_python(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj

def grade_task1(df: pd.DataFrame, expected: Dict, step_number: int) -> Tuple[float, Dict]:
    scores = {}
    total_cols = ["age", "gender", "blood_pressure", "cholesterol"]

    for col in total_cols:
        if col not in df.columns:
            scores[col] = 0.001
            continue
        null_count = df[col].isnull().sum()
        original_nulls = expected["total_nulls"] / len(total_cols)
        if original_nulls == 0:
            scores[col] = 0.999
        else:
            scores[col] = clamp(1.0 - (null_count / original_nulls))

    null_score = clamp(sum(scores.values()) / len(scores))
    efficiency_penalty = round(min(0.2, step_number * 0.01), 3)
    final_score = clamp(null_score - efficiency_penalty)

    info = {
        "column_scores": scores,
        "null_score": null_score,
        "efficiency_penalty": efficiency_penalty,
        "remaining_nulls": int(df.isnull().sum().sum())
    }

    return final_score, to_python(info)


def grade_task2(df: pd.DataFrame, expected: Dict, step_number: int) -> Tuple[float, Dict]:
    type_map = {
        "amount":      (float, "float64"),
        "quantity":    (int,   "int64"),
        "is_returned": (bool,  "bool"),
        "rating":      (float, "float64"),
    }

    type_scores = {}
    for col, (py_type, np_type) in type_map.items():
        if col not in df.columns:
            type_scores[col] = 0.001
            continue
        actual = str(df[col].dtype)
        if np_type in actual or (np_type == "bool" and actual == "bool"):
            type_scores[col] = 0.999
        elif "float" in actual and np_type == "float64":
            type_scores[col] = 0.999
        elif "int" in actual and np_type == "int64":
            type_scores[col] = 0.999
        else:
            type_scores[col] = 0.001

    dtype_score = clamp(sum(type_scores.values()) / len(type_scores))

    remaining_dupes  = int(df.duplicated().sum())
    original_dupes   = expected["duplicate_count"]
    duplicate_score  = clamp(1.0 - (remaining_dupes / original_dupes)) if original_dupes > 0 else 0.999

    combined = clamp(0.6 * dtype_score + 0.4 * duplicate_score)
    efficiency_penalty = round(min(0.2, step_number * 0.008), 3)
    final_score = clamp(combined - efficiency_penalty)

    info = {
        "type_scores":        type_scores,
        "dtype_score":        dtype_score,
        "duplicate_score":    duplicate_score,
        "remaining_dupes":    remaining_dupes,
        "efficiency_penalty": efficiency_penalty,
    }

    return final_score, to_python(info)


def grade_task3(df: pd.DataFrame, expected: Dict, step_number: int) -> Tuple[float, Dict]:
    null_cols = ["temperature", "humidity", "pressure", "sensor_type"]
    null_remaining = sum(df[c].isnull().sum() for c in null_cols if c in df.columns)
    null_score = clamp(1.0 - (null_remaining / 20))

    dtype_score = 0.999 if str(df["battery_level"].dtype) in ["float64", "float32"] else 0.001

    remaining_dupes = int(df.duplicated().sum())
    duplicate_score = clamp(1.0 - (remaining_dupes / 4))

    bounds = expected["outlier_bounds"]
    outlier_scores = {}
    for col, (low, high) in bounds.items():
        if col not in df.columns:
            outlier_scores[col] = 0.001
            continue
        clean = df[col].dropna()
        violations = int(((clean < low) | (clean > high)).sum())
        outlier_scores[col] = clamp(1.0 - (violations / len(clean))) if len(clean) > 0 else 0.001

    outlier_score = clamp(sum(outlier_scores.values()) / len(outlier_scores))

    if "status" in df.columns:
        unique_vals = df["status"].dropna().unique()
        is_standardized = all(v == v.lower() for v in unique_vals)
        category_score = 0.999 if is_standardized else 0.001
    else:
        category_score = 0.001

    combined = clamp(
        0.25 * null_score +
        0.15 * dtype_score +
        0.20 * duplicate_score +
        0.25 * outlier_score +
        0.15 * category_score
    )

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

    return final_score, to_python(info)


GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}
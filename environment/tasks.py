import pandas as pd
import numpy as np
from typing import Dict, Tuple


def get_task1_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Easy task — Patient intake form with missing values only.
    Agent must fill nulls in all columns correctly.
    """
    np.random.seed(42)
    n = 50

    df = pd.DataFrame({
        "patient_id": range(1001, 1001 + n),
        "age": [25, 34, None, 45, 52, None, 38, 29, None, 61,
                44, None, 33, 57, 48, None, 72, 26, 39, None,
                55, 41, None, 63, 37, 28, None, 50, 44, 36,
                None, 58, 31, 47, None, 65, 42, None, 53, 27,
                46, None, 35, 60, 43, None, 38, 51, None, 40],
        "gender": ["Male", "Female", None, "Male", "Female", "Male", None,
                   "Female", "Male", "Female", None, "Male", "Female", None,
                   "Male", "Female", "Male", None, "Female", "Male",
                   "Female", "Male", None, "Female", "Male", "Female",
                   "Male", None, "Female", "Male", "Female", None,
                   "Male", "Female", "Male", "Female", None, "Male",
                   "Female", "Male", None, "Female", "Male", "Female",
                   "Male", None, "Female", "Male", "Female", None],
        "blood_pressure": [120, None, 135, 118, None, 142, 128, None,
                           115, 138, None, 125, 132, None, 119, 145,
                           None, 122, 136, None, 129, 118, None, 141,
                           127, None, 133, 121, None, 139, 126, None,
                           117, 143, None, 124, 131, None, 120, 137,
                           None, 128, 115, None, 142, 126, None, 134,
                           122, None],
        "cholesterol": [180, 220, 195, None, 240, 175, 210, 230, None,
                        185, 215, 200, None, 245, 170, 205, 225, None,
                        190, 235, 180, None, 210, 195, None, 240, 175,
                        220, None, 185, 215, 200, None, 245, 170, 205,
                        None, 225, 190, 235, 180, None, 210, 195, 240,
                        None, 175, 220, 185, None],
    })

    expected = {
        "fill_strategy": {
            "age": "median",
            "gender": "mode",
            "blood_pressure": "median",
            "cholesterol": "median"
        },
        "total_nulls": int(df.isnull().sum().sum())
    }

    return df, expected


def get_task2_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Medium task — E-commerce orders with type errors + duplicates.
    Agent must cast types and drop duplicate rows.
    """
    np.random.seed(42)
    n = 60

    df = pd.DataFrame({
        "order_id": list(range(5001, 5001 + n)),
        "customer_id": [f"C{str(i).zfill(4)}" for i in range(1, n + 1)],
        "order_date": ["2024-01-15", "2024-02-20", "2024-03-10",
                       "2024-01-22", "2024-02-14"] * 12,
        "amount": [str(round(100 + i * 2.5, 2)) for i in range(n)],
        "quantity": [str(i % 10 + 1) for i in range(n)],
        "is_returned": ["True", "False", "True", "False", "False",
                        "True", "False", "True", "False", "True"] * 6,
        "rating": [str(round(3.0 + (i % 5) * 0.5, 1)) for i in range(n)],
    })

    # inject duplicates
    duplicate_rows = df.iloc[[2, 7, 15, 23, 31]].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    expected = {
        "type_casts": {
            "amount": "float",
            "quantity": "int",
            "is_returned": "bool",
            "rating": "float",
            "order_date": "datetime"
        },
        "duplicate_count": 5
    }

    return df, expected


def get_task3_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Hard task — Sensor log with nulls + type errors + duplicates + outliers
    + inconsistent categories. Agent must handle all issues.
    """
    np.random.seed(42)
    n = 80

    temperatures = [round(20 + np.random.normal(0, 3), 2) for _ in range(n)]
    temperatures[5]  = 999.0
    temperatures[18] = -150.0
    temperatures[42] = 850.0
    temperatures[67] = -200.0

    humidity = [round(50 + np.random.normal(0, 10), 2) for _ in range(n)]
    humidity[10] = 500.0
    humidity[35] = -50.0
    humidity[55] = 300.0

    pressure = [round(1013 + np.random.normal(0, 5), 2) for _ in range(n)]
    pressure[20] = 9999.0
    pressure[60] = -999.0

    df = pd.DataFrame({
        "sensor_id": [f"S{str(i).zfill(3)}" for i in range(1, n + 1)],
        "timestamp":  ["2024-01-15 08:00:00"] * n,
        "temperature": temperatures,
        "humidity":    humidity,
        "pressure":    pressure,
        "status": (["active", "Active", "ACTIVE", "inactive",
                    "Inactive", "maintenance", "Maintenance", "MAINTENANCE"] * 10)[:n],
        "sensor_type": (["temperature", "humidity", "pressure",
                         None, "temperature", None, "humidity", "pressure"] * 10)[:n],
        "battery_level": [str(round(20 + (i % 80), 1)) for i in range(n)],
    })

    # inject nulls
    null_indices = [3, 8, 14, 21, 29, 36, 44, 52, 63, 71]
    df.loc[null_indices, "temperature"] = None
    df.loc[null_indices[:6], "humidity"] = None
    df.loc[null_indices[:4], "pressure"] = None

    # inject duplicates
    duplicate_rows = df.iloc[[1, 12, 25, 40]].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    expected = {
        "fill_strategy": {
            "temperature": "median",
            "humidity": "median",
            "pressure": "median",
            "sensor_type": "mode"
        },
        "type_casts": {
            "battery_level": "float",
            "timestamp": "datetime"
        },
        "duplicate_count": 4,
        "outlier_bounds": {
            "temperature": (-50, 60),
            "humidity": (0, 100),
            "pressure": (900, 1100)
        },
        "category_standardization": {
            "status": "lowercase"
        }
    }

    return df, expected


TASKS = {
    "task1": get_task1_data,
    "task2": get_task2_data,
    "task3": get_task3_data,
}
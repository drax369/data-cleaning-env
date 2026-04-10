import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
from environment.models import Observation, Action, Reward, EnvironmentState
from environment.tasks import TASKS
from environment.graders import GRADERS


MAX_STEPS = 50


class DataCleaningEnv:

    def __init__(self, task_id: str = "task1"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Choose from {list(TASKS.keys())}")
        self.task_id    = task_id
        self.df         = None
        self.expected   = None
        self.step_count = 0
        self.done       = False
        self.current_score = 0.0

    # ------------------------------------------------------------------ #
    #  reset()                                                             #
    # ------------------------------------------------------------------ #
    def reset(self) -> Observation:
        df_raw, expected  = TASKS[self.task_id]()
        self.df           = df_raw.copy()
        self.expected     = expected
        self.step_count   = 0
        self.done         = False
        self.current_score = 0.001
        return self._make_observation("Environment reset. Ready for cleaning.")

    # ------------------------------------------------------------------ #
    #  step()                                                              #
    # ------------------------------------------------------------------ #
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            obs = self._make_observation("Episode already finished. Call reset().")
            reward = self._make_reward(0.001, {})
            return obs, reward, True, {"error": "Episode done"}

        if self.df is None:
            raise RuntimeError("Call reset() before step()")

        self.step_count += 1
        info = {}

        try:
            info = self._apply_action(action)
        except Exception as e:
            info = {"error": str(e), "action_result": f"Action failed: {str(e)}"}

        try:
            score, grade_info = GRADERS[self.task_id](
                self.df, self.expected, self.step_count
            )
        except Exception as e:
            score = 0.001
            grade_info = {"error": str(e)}

        self.current_score = max(0.001, min(0.999, float(score)))
        info.update(grade_info)

        if self.current_score >= 0.95 or self.step_count >= MAX_STEPS:
            self.done = True

        obs    = self._make_observation(info.get("action_result", ""))
        reward = self._make_reward(self.current_score, grade_info)
        return obs, reward, self.done, info
    # ------------------------------------------------------------------ #
    #  state()                                                             #
    # ------------------------------------------------------------------ #
    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task_id       = self.task_id,
            step_number   = self.step_count,
            max_steps     = MAX_STEPS,
            done          = self.done,
            current_score = max(0.001, min(0.999, self.current_score)),
            dataset_shape = list(self.df.shape) if self.df is not None else [0, 0],
    )

    # ------------------------------------------------------------------ #
    #  action dispatcher                                                   #
    # ------------------------------------------------------------------ #
    def _apply_action(self, action: Action) -> Dict:
        at = action.action_type

        if at == "fill_null":
            return self._fill_null(action)
        elif at == "cast_type":
            return self._cast_type(action)
        elif at == "drop_duplicates":
            return self._drop_duplicates()
        elif at == "clip_outliers":
            return self._clip_outliers(action)
        elif at == "standardize_categories":
            return self._standardize_categories(action)
        else:
            return {"action_result": f"Unknown action: {at}"}

    # ------------------------------------------------------------------ #
    #  individual actions                                                  #
    # ------------------------------------------------------------------ #
    def _fill_null(self, action: Action) -> Dict:
        col = action.column
        if col is None or col not in self.df.columns:
            return {"action_result": f"Column '{col}' not found — skipped"}

        before = int(self.df[col].isnull().sum())
        if before == 0:
            return {"action_result": f"No nulls in '{col}'"}

        method = action.method or "median"
        value  = action.value

        try:
            if value is not None:
                self.df[col] = self.df[col].fillna(value)
            elif method == "median":
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif method == "mean":
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif method == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif method == "ffill":
                self.df[col] = self.df[col].ffill()
            elif method == "bfill":
                self.df[col] = self.df[col].bfill()
            else:
                return {"action_result": f"Unknown fill method: {method}"}
        except Exception as e:
            return {"action_result": f"Fill failed on '{col}': {str(e)}"}

        after = int(self.df[col].isnull().sum())
        return {"action_result": f"Filled {before - after} nulls in '{col}' using {method}"}

    def _cast_type(self, action: Action) -> Dict:
        col   = action.column
        dtype = action.dtype
        if col not in self.df.columns:
            return {"action_result": f"Column '{col}' not found"}

        before = str(self.df[col].dtype)
        try:
            if dtype == "float":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype(float)
            elif dtype == "int":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype("Int64")
            elif dtype == "bool":
                self.df[col] = self.df[col].map(
                    {"True": True, "False": False, True: True, False: False}
                )
            elif dtype == "datetime":
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            elif dtype == "str":
                self.df[col] = self.df[col].astype(str)
            else:
                return {"action_result": f"Unknown dtype: {dtype}"}
        except Exception as e:
            return {"action_result": f"Cast failed: {str(e)}"}

        return {"action_result": f"Cast '{col}' from {before} to {dtype}"}

    def _drop_duplicates(self) -> Dict:
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        after   = len(self.df)
        return {"action_result": f"Dropped {before - after} duplicate rows"}

    def _clip_outliers(self, action: Action) -> Dict:
        col = action.column
        if col not in self.df.columns:
            return {"action_result": f"Column '{col}' not found"}

        if action.value and isinstance(action.value, dict):
            low  = action.value.get("low")
            high = action.value.get("high")
        else:
            q1  = self.df[col].quantile(0.25)
            q3  = self.df[col].quantile(0.75)
            iqr = q3 - q1
            low  = q1 - 3 * iqr
            high = q3 + 3 * iqr

        before = int(((self.df[col] < low) | (self.df[col] > high)).sum())
        self.df[col] = self.df[col].clip(lower=low, upper=high)
        return {"action_result": f"Clipped {before} outliers in '{col}' to [{low}, {high}]"}

    def _standardize_categories(self, action: Action) -> Dict:
        col     = action.column
        mapping = action.mapping

        if col not in self.df.columns:
            return {"action_result": f"Column '{col}' not found"}

        if mapping:
            self.df[col] = self.df[col].map(mapping).fillna(self.df[col])
        else:
            self.df[col] = self.df[col].str.lower().str.strip()

        return {"action_result": f"Standardized categories in '{col}'"}

    # ------------------------------------------------------------------ #
    #  helpers                                                             #
    # ------------------------------------------------------------------ #
    def _make_observation(self, message: str) -> Observation:
        df = self.df

        null_counts = {
            col: int(df[col].isnull().sum())
            for col in df.columns
        }

        dtype_issues = {}
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(3).tolist()
                dtype_issues[col] = f"object — sample: {sample}"

        outlier_counts = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            q1  = df[col].quantile(0.25)
            q3  = df[col].quantile(0.75)
            iqr = q3 - q1
            low  = q1 - 3 * iqr
            high = q3 + 3 * iqr
            count = int(((df[col] < low) | (df[col] > high)).sum())
            if count > 0:
                outlier_counts[col] = count

        sample_rows = df.head(3).fillna("NULL").astype(str).to_dict(orient="records")

        return Observation(
            task_id         = self.task_id,
            step_number     = self.step_count,
            dataset_shape   = list(df.shape),
            null_counts     = null_counts,
            dtype_issues    = dtype_issues,
            duplicate_count = int(df.duplicated().sum()),
            outlier_counts  = outlier_counts,
            columns         = list(df.columns),
            sample_rows     = sample_rows,
            message         = message,
        )

    def _make_reward(self, score: float, grade_info: Dict) -> Reward:
        def c(v):
            return float(max(0.001, min(0.999, v)))
        return Reward(
            score              = c(score),
            null_score         = c(grade_info.get("null_score", 0.001)),
            dtype_score        = c(grade_info.get("dtype_score", 0.001)),
            duplicate_score    = c(grade_info.get("duplicate_score", 0.001)),
            outlier_score      = c(grade_info.get("outlier_score", 0.001)),
            efficiency_penalty = c(grade_info.get("efficiency_penalty", 0.001)),
            done               = self.done,
            info               = grade_info,
    )
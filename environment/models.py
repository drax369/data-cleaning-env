from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Observation(BaseModel):
    task_id: str
    step_number: int
    dataset_shape: List[int]
    null_counts: Dict[str, int]
    dtype_issues: Dict[str, str]
    duplicate_count: int
    outlier_counts: Dict[str, int]
    columns: List[str]
    sample_rows: List[Dict[str, Any]]
    message: str = ""


class Action(BaseModel):
    action_type: str
    column: Optional[str] = None
    value: Optional[Any] = None
    dtype: Optional[str] = None
    mapping: Optional[Dict[str, Any]] = None
    method: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    null_score: float = Field(ge=0.0, le=1.0)
    dtype_score: float = Field(ge=0.0, le=1.0)
    duplicate_score: float = Field(ge=0.0, le=1.0)
    outlier_score: float = Field(ge=0.0, le=1.0)
    efficiency_penalty: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = {}


class EnvironmentState(BaseModel):
    task_id: str
    step_number: int
    max_steps: int
    done: bool
    current_score: float = Field(ge=0.0, le=1.0)
    dataset_shape: List[int]
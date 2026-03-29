title: Data Cleaning Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
# Data Cleaning OpenEnv

A real-world [OpenEnv](https://openenv.dev) environment where AI agents learn to clean messy datasets.
Built for the Meta AI Hackathon.

## What it simulates

A data analyst receiving dirty CSV files and cleaning them systematically.
Issues include missing values, wrong data types, duplicate rows, outliers,
and inconsistent category labels — problems found in every real-world dataset.

## Tasks

| Task | Domain | Difficulty | Issues |
|------|--------|------------|--------|
| task1 | Hospital patient intake | Easy | Missing values only |
| task2 | E-commerce orders export | Medium | Wrong types + duplicates |
| task3 | IoT sensor log | Hard | Nulls + types + dupes + outliers + categories |

## Action Space

| Action | Parameters | Description |
|--------|------------|-------------|
| `fill_null` | `column`, `method` (median/mean/mode/ffill/bfill) | Fill missing values |
| `cast_type` | `column`, `dtype` (float/int/bool/datetime/str) | Fix data types |
| `drop_duplicates` | — | Remove duplicate rows |
| `clip_outliers` | `column`, `value` ({low, high}) | Clip extreme values |
| `standardize_categories` | `column`, `mapping` (optional) | Normalize text categories |

## Observation Space
```json
{
  "task_id": "task1",
  "step_number": 3,
  "dataset_shape": [50, 5],
  "null_counts": {"age": 0, "gender": 12, "blood_pressure": 17, "cholesterol": 12},
  "dtype_issues": {},
  "duplicate_count": 0,
  "outlier_counts": {},
  "columns": ["patient_id", "age", "gender", "blood_pressure", "cholesterol"],
  "sample_rows": [...],
  "message": "Filled 14 nulls in 'age' using median"
}
```

## Reward Function

Scores range from 0.0 to 1.0 with partial credit at every step.

| Component | Weight (task3) | Description |
|-----------|---------------|-------------|
| null_score | 25% | Proportion of nulls resolved |
| dtype_score | 15% | Proportion of type issues fixed |
| duplicate_score | 20% | Proportion of duplicates removed |
| outlier_score | 25% | Proportion of outliers clipped |
| category_score | 15% | Categories standardized |
| efficiency_penalty | -0–20% | Penalty for excessive steps |

## Baseline Scores

| Task | Baseline Score | Difficulty |
|------|---------------|------------|
| task1 | ~0.85 | Easy |
| task2 | ~0.70 | Medium |
| task3 | ~0.55 | Hard |

## Setup
```bash
git clone https://github.com/drax369/data-cleaning-env.git
cd data-cleaning-env
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run the API server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Visit http://localhost:7860/docs for the interactive API explorer.

## Run baseline inference
```bash
export OPENAI_API_KEY=your_key_here
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

## Docker
```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_key \
  -e MODEL_NAME=gpt-4o-mini \
  data-cleaning-env
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset/{task_id}` | Reset environment, get first observation |
| POST | `/step/{task_id}` | Submit action, get observation + reward |
| GET | `/state/{task_id}` | Get current environment state |
| GET | `/openenv.yaml` | OpenEnv spec metadata |

## Episode Example
```python
from environment.env import DataCleaningEnv
from environment.models import Action

env = DataCleaningEnv("task1")
obs = env.reset()

action = Action(action_type="fill_null", column="age", method="median")
obs, reward, done, info = env.step(action)
print(reward.score)  # 0.304
```

## Project Structure
```
data-cleaning-env/
├── environment/
│   ├── env.py        # core step/reset/state logic
│   ├── tasks.py      # 3 task definitions + dirty datasets
│   ├── graders.py    # scoring functions
│   └── models.py     # Pydantic data models
├── app.py            # FastAPI server
├── inference.py      # baseline agent script
├── openenv.yaml      # OpenEnv spec
├── Dockerfile        # container definition
├── requirements.txt  # Python dependencies
└── README.md
```
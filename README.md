---
title: Data Cleaning Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
license: mit
tags:
  - openenv
---
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

## Known limitations & future work

### Imputation bias on imbalanced categorical columns
The current environment trains the agent to fill null categorical values
using mode (most frequent value). This works well for balanced columns
but introduces **imputation bias** on imbalanced ones.

**Example:** A gender column with 80% male and 20% female entries will
cause the agent to fill every missing gender as "Male" — statistically
defensible but factually incorrect and harmful in health or demographic
datasets where the null itself may carry meaning (e.g. the person chose
not to disclose).

**What a production system should do instead:**
- Flag nulls in sensitive columns (gender, race, income) as `"Unknown"`
  rather than imputing them
- Detect column imbalance before choosing a fill strategy
- Treat the null as meaningful data in survey or opt-in contexts

This is a known open problem in automated data cleaning and a natural
direction for future work.

---

### Duplicate removal preserves one copy
`drop_duplicates()` keeps the first occurrence of a duplicated row and
removes all subsequent ones. So 5 identical rows become 1 — no data is
lost, only redundancy is removed. This is correct behavior for accidental
duplicates (e.g. a form submitted twice) but may be inappropriate when
duplicates carry legitimate meaning (e.g. two patients with identical
intake data who are genuinely different people).

A future improvement would be to expose a `keep` parameter in the action
space (`first`, `last`, or `none`) so the agent can choose the right
strategy per context.

---

### Outlier clipping vs removal
The current environment clips outliers to boundary values rather than
removing the row. This preserves dataset size but may introduce
artificial boundary clustering. Future work could add a
`drop_outliers` action alongside `clip_outliers`.

---

### Static task difficulty
The three tasks have fixed schemas and fixed dirty patterns. A more
robust environment would procedurally generate dirty datasets with
varying column types, imbalance ratios, and issue combinations so the
agent cannot overfit to the specific task structure.

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
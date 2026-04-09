import os
import json
from openai import OpenAI
from environment.env import DataCleaningEnv
from environment.models import Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")

try:
    client = OpenAI(
        api_key  = HF_TOKEN or "dummy-key",
        base_url = API_BASE_URL,
    )
except Exception as e:
    print(f"[END] task=init score=0.0 steps=0 error=client_init_failed", flush=True)
    raise

MAX_STEPS = 20


def build_prompt(obs, task_id: str) -> str:
    return f"""You are a data cleaning agent. Your job is to clean a messy dataset.

TASK: {task_id}
STEP: {obs.step_number}
DATASET SHAPE: {obs.dataset_shape[0]} rows x {obs.dataset_shape[1]} columns
COLUMNS: {obs.columns}

CURRENT ISSUES:
- Null counts per column: {json.dumps(obs.null_counts)}
- Dtype issues (object columns): {json.dumps(obs.dtype_issues)}
- Duplicate rows: {obs.duplicate_count}
- Outlier counts: {json.dumps(obs.outlier_counts)}

SAMPLE ROWS (first 3):
{json.dumps(obs.sample_rows, indent=2)}

LAST MESSAGE: {obs.message}

AVAILABLE ACTIONS:
1. fill_null       - {{"action_type": "fill_null", "column": "<col>", "method": "median|mean|mode|ffill|bfill"}}
2. cast_type       - {{"action_type": "cast_type", "column": "<col>", "dtype": "float|int|bool|datetime|str"}}
3. drop_duplicates - {{"action_type": "drop_duplicates"}}
4. clip_outliers   - {{"action_type": "clip_outliers", "column": "<col>", "value": {{"low": <n>, "high": <n>}}}}
5. standardize_categories - {{"action_type": "standardize_categories", "column": "<col>"}}

Respond with ONLY a JSON object for the single best next action.
If all issues are resolved respond with: {{"action_type": "done"}}
"""


def run_task(task_id: str) -> float:
    print(f"[START] task={task_id}", flush=True)

    env = DataCleaningEnv(task_id)
    obs = env.reset()
    final_score = 0.0
    step_count  = 0

    for step in range(MAX_STEPS):
        prompt = build_prompt(obs, task_id)

        try:
            response = client.chat.completions.create(
                model    = MODEL_NAME,
                messages = [
                    {"role": "system", "content": "You are a data cleaning agent. Always respond with valid JSON only."},
                    {"role": "user",   "content": prompt}
                ],
                temperature = 0.0,
                max_tokens  = 200,
            )
            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            action_dict = json.loads(raw)

        except json.JSONDecodeError:
            print(f"[STEP] task={task_id} step={step+1} reward={final_score:.3f} error=json_parse", flush=True)
            continue
        except Exception as e:
            print(f"[STEP] task={task_id} step={step+1} reward={final_score:.3f} error=api_error", flush=True)
            break

        if action_dict.get("action_type") == "done":
            print(f"[STEP] task={task_id} step={step+1} reward={final_score:.3f} action=done", flush=True)
            break

        try:
            action = Action(**action_dict)
            obs, reward, done, info = env.step(action)
            final_score = reward.score
            step_count  = step + 1
        except Exception as e:
            print(f"[STEP] task={task_id} step={step+1} reward={final_score:.3f} error=step_failed", flush=True)
            continue

        print(f"[STEP] task={task_id} step={step+1} reward={final_score:.3f} action={action_dict.get('action_type')} column={action_dict.get('column', 'N/A')}", flush=True)

        if done:
            step_count = step + 1
            break

    print(f"[END] task={task_id} score={final_score:.3f} steps={step_count}", flush=True)
    return final_score


if __name__ == "__main__":
    print(f"[START] task=inference model={MODEL_NAME}", flush=True)

    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        scores[task_id] = run_task(task_id)

    avg = sum(scores.values()) / len(scores)
    for task_id, score in scores.items():
        print(f"[STEP] task=results {task_id}={score:.3f}", flush=True)

    print(f"[END] task=inference score={avg:.3f} steps={MAX_STEPS}", flush=True)
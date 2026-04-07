import os
import json
from openai import OpenAI
from environment.env import DataCleaningEnv
from environment.models import Action

# ------------------------------------------------------------------ #
#  credentials — read from environment variables                       #
# ------------------------------------------------------------------ #
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")

client = OpenAI(
    api_key  = HF_TOKEN if HF_TOKEN else os.environ.get("OPENAI_API_KEY", ""),
    base_url = API_BASE_URL,
)

MAX_STEPS = 20  # keep inference fast — well within 20 min limit


# ------------------------------------------------------------------ #
#  prompt builder                                                      #
# ------------------------------------------------------------------ #
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
1. fill_null       — {{"action_type": "fill_null", "column": "<col>", "method": "median|mean|mode|ffill|bfill"}}
2. cast_type       — {{"action_type": "cast_type", "column": "<col>", "dtype": "float|int|bool|datetime|str"}}
3. drop_duplicates — {{"action_type": "drop_duplicates"}}
4. clip_outliers   — {{"action_type": "clip_outliers", "column": "<col>", "value": {{"low": <n>, "high": <n>}}}}
5. standardize_categories — {{"action_type": "standardize_categories", "column": "<col>"}}

Respond with ONLY a JSON object for the single best next action.
Pick the most impactful action based on the current issues.
If all issues are resolved respond with: {{"action_type": "done"}}
"""


# ------------------------------------------------------------------ #
#  single task runner                                                  #
# ------------------------------------------------------------------ #
def run_task(task_id: str) -> float:
    print(f"START {task_id}")

    env = DataCleaningEnv(task_id)
    obs = env.reset()
    final_score = 0.0

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

            # strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            action_dict = json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"STEP {task_id} step={step+1} error=json_parse_error")
            continue
        except Exception as e:
            print(f"STEP {task_id} step={step+1} error=api_error")
            break

        # agent signals it is done
        if action_dict.get("action_type") == "done":
            print(f"STEP {task_id} step={step+1} action=done score={final_score}")
            break

        action = Action(**action_dict)
        obs, reward, done, info = env.step(action)
        final_score = reward.score

        print(f"STEP {task_id} step={step+1} action={action_dict.get('action_type')} column={action_dict.get('column', 'N/A')} score={final_score:.3f}")

        if done:
            break

    print(f"END {task_id} score={final_score:.3f}")
    return final_score


# ------------------------------------------------------------------ #
#  main                                                                #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("START inference")
    print(f"STEP config model={MODEL_NAME} base_url={API_BASE_URL}")

    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        scores[task_id] = run_task(task_id)

    print("STEP results " + " ".join([f"{k}={v:.3f}" for k, v in scores.items()]))
    avg = sum(scores.values()) / len(scores)
    print(f"END inference avg_score={avg:.3f}")
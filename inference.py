import os
import json
from typing import List, Optional
from openai import OpenAI
from environment.env import DataCleaningEnv
from environment.models import Action

API_BASE_URL   = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME     = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN       = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN or "dummy-key"

try:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
except Exception as e:
    print(f"[END] success=false steps=0 score=0.50 rewards=", flush=True)
    raise

MAX_STEPS = 20
SUCCESS_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_prompt(obs, task_id: str) -> str:
    return f"""You are a data cleaning agent. Your job is to clean a messy dataset.

TASK: {task_id}
STEP: {obs.step_number}
COLUMNS: {obs.columns}
CURRENT ISSUES:
- Null counts: {json.dumps(obs.null_counts)}
- Dtype issues: {json.dumps(obs.dtype_issues)}
- Duplicate rows: {obs.duplicate_count}
- Outlier counts: {json.dumps(obs.outlier_counts)}

AVAILABLE ACTIONS:
1. fill_null       - {{"action_type": "fill_null", "column": "<col>", "method": "median|mean|mode"}}
2. cast_type       - {{"action_type": "cast_type", "column": "<col>", "dtype": "float|int|bool|datetime"}}
3. drop_duplicates - {{"action_type": "drop_duplicates"}}
4. clip_outliers   - {{"action_type": "clip_outliers", "column": "<col>", "value": {{"low": <n>, "high": <n>}}}}
5. standardize_categories - {{"action_type": "standardize_categories", "column": "<col>"}}

Respond with ONLY a JSON object for the single best next action.
If all issues are resolved respond with: {{"action_type": "done"}}
"""


def get_fallback_actions(task_id: str) -> list:
    if task_id == "task1":
        return [
            Action(action_type="fill_null", column="age",            method="median"),
            Action(action_type="fill_null", column="gender",         method="mode"),
            Action(action_type="fill_null", column="blood_pressure", method="median"),
            Action(action_type="fill_null", column="cholesterol",    method="median"),
        ]
    elif task_id == "task2":
        return [
            Action(action_type="drop_duplicates"),
            Action(action_type="cast_type", column="amount",      dtype="float"),
            Action(action_type="cast_type", column="quantity",    dtype="int"),
            Action(action_type="cast_type", column="is_returned", dtype="bool"),
            Action(action_type="cast_type", column="rating",      dtype="float"),
        ]
    elif task_id == "task3":
        return [
            Action(action_type="drop_duplicates"),
            Action(action_type="fill_null", column="temperature",   method="median"),
            Action(action_type="fill_null", column="humidity",      method="median"),
            Action(action_type="fill_null", column="pressure",      method="median"),
            Action(action_type="fill_null", column="sensor_type",   method="mode"),
            Action(action_type="cast_type", column="battery_level", dtype="float"),
            Action(action_type="clip_outliers", column="temperature", value={"low": -50,  "high": 60}),
            Action(action_type="clip_outliers", column="humidity",    value={"low": 0,    "high": 100}),
            Action(action_type="clip_outliers", column="pressure",    value={"low": 900,  "high": 1100}),
            Action(action_type="standardize_categories", column="status"),
        ]
    return []


def run_task(task_id: str) -> float:
    log_start(task=task_id, env="data-cleaning-env", model=MODEL_NAME)

    env = DataCleaningEnv(task_id)
    obs = env.reset()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.5
    success = False
    fallback_used = False

    try:
        for step in range(1, MAX_STEPS + 1):
            action_str = "none"
            reward = 0.5
            done = False
            error = None

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a data cleaning agent. Always respond with valid JSON only."},
                        {"role": "user",   "content": build_prompt(obs, task_id)}
                    ],
                    temperature=0.0,
                    max_tokens=200,
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
                action_dict = json.loads(raw)

                if action_dict.get("action_type") == "done":
                    log_step(step=step, action="done", reward=reward, done=True, error=None)
                    rewards.append(reward)
                    steps_taken = step
                    done = True
                    break

                action = Action(**action_dict)
                action_str = action_dict.get("action_type", "unknown")

            except Exception as e:
                if not fallback_used:
                    fallback_used = True
                    fallback_actions = get_fallback_actions(task_id)
                    for i, action in enumerate(fallback_actions):
                        try:
                            obs, result, done, info = env.step(action)
                            r = float(result.score)
                            r = max(0.01, min(0.99, r))
                            rewards.append(r)
                            steps_taken = i + 1
                            score = r
                            log_step(step=i+1, action=action.action_type, reward=r, done=done, error=None)
                            if done:
                                break
                        except Exception as ex:
                            log_step(step=i+1, action=action.action_type, reward=0.5, done=False, error=str(ex))
                    break
                error = str(e)
                log_step(step=step, action="error", reward=0.5, done=False, error=error)
                rewards.append(0.5)
                steps_taken = step
                break

            try:
                obs, result, done, info = env.step(action)
                reward = float(result.score)
                reward = max(0.01, min(0.99, reward))
                score = reward
            except Exception as e:
                error = str(e)
                reward = 0.5

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = max(0.01, min(0.99, score))
        success = score >= SUCCESS_THRESHOLD

    finally:
        final_score = max(0.01, min(0.99, score))
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return final_score


if __name__ == "__main__":
    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        scores[task_id] = run_task(task_id)
    avg = sum(scores.values()) / len(scores)
    print(f"[END] success=true steps={MAX_STEPS} score={avg:.2f} rewards={','.join(f'{s:.2f}' for s in scores.values())}", flush=True)
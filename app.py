import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from environment.env import DataCleaningEnv
from environment.models import Action
from typing import Dict
import json

app = FastAPI(
    title       = "Data Cleaning OpenEnv",
    description = "Real-world OpenEnv environment for AI data cleaning agents",
    version     = "1.0.0",
)

envs: Dict[str, DataCleaningEnv] = {
    "task1": DataCleaningEnv("task1"),
    "task2": DataCleaningEnv("task2"),
    "task3": DataCleaningEnv("task3"),
}


def clean(obj):
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@app.get("/")
def root():
    return {"status": "ok", "name": "data-cleaning-env", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset_default():
    """OpenEnv standard reset — defaults to task1"""
    obs = envs["task1"].reset()
    return JSONResponse(content=clean(obs.model_dump()))


@app.post("/step")
def step_default(action: Action):
    """OpenEnv standard step — defaults to task1"""
    env = envs["task1"]
    if env.df is None:
        env.reset()
    try:
        obs, reward, done, info = env.step(action)
        return JSONResponse(content=clean({
            "observation": obs.model_dump(),
            "reward":      reward.model_dump(),
            "done":        done,
            "info":        info,
        }))
    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state_default():
    """OpenEnv standard state — defaults to task1"""
    env = envs["task1"]
    if env.df is None:
        env.reset()
    return JSONResponse(content=clean(env.state().model_dump()))

@app.get("/openenv.yaml")
def get_spec():
    with open("openenv.yaml", "r") as f:
        content = f.read()
    return JSONResponse(content={"spec": content})


@app.post("/reset/{task_id}")
def reset(task_id: str):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    obs = envs[task_id].reset()
    return JSONResponse(content=clean(obs.model_dump()))


@app.post("/step/{task_id}")
def step(task_id: str, action: Action):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    env = envs[task_id]
    if env.df is None:
        raise HTTPException(status_code=400, detail="Call /reset/{task_id} first")
    try:
        obs, reward, done, info = env.step(action)
        result = clean({
            "observation": obs.model_dump(),
            "reward":      reward.model_dump(),
            "done":        done,
            "info":        info,
        })
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"STEP ERROR task={task_id}: {tb}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state/{task_id}")
def state(task_id: str):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    env = envs[task_id]
    if env.df is None:
        raise HTTPException(status_code=400, detail="Call /reset/{task_id} first")
    return JSONResponse(content=clean(envs[task_id].state().model_dump()))


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "task1", "name": "Patient Intake Null Imputation",   "difficulty": "easy"},
            {"id": "task2", "name": "E-Commerce Order Type Fixing",     "difficulty": "medium"},
            {"id": "task3", "name": "IoT Sensor Log Full Pipeline",     "difficulty": "hard"},
        ]
    }
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from environment.env import DataCleaningEnv
from environment.models import Action
from typing import Dict
import yaml

app = FastAPI(
    title       = "Data Cleaning OpenEnv",
    description = "Real-world OpenEnv environment for AI data cleaning agents",
    version     = "1.0.0",
)

# one environment instance per task — kept in memory
envs: Dict[str, DataCleaningEnv] = {
    "task1": DataCleaningEnv("task1"),
    "task2": DataCleaningEnv("task2"),
    "task3": DataCleaningEnv("task3"),
}


# ------------------------------------------------------------------ #
#  health check — hackathon checker pings this                        #
# ------------------------------------------------------------------ #
@app.get("/")
def root():
    return {"status": "ok", "name": "data-cleaning-env", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------------ #
#  openenv spec                                                        #
# ------------------------------------------------------------------ #
@app.get("/openenv.yaml")
def get_spec():
    with open("openenv.yaml", "r") as f:
        content = f.read()
    return JSONResponse(content={"spec": content})


# ------------------------------------------------------------------ #
#  core API endpoints                                                  #
# ------------------------------------------------------------------ #
@app.post("/reset")
def reset_all():
    results = {}
    for task_id, env in envs.items():
        obs = env.reset()
        results[task_id] = obs.model_dump()
    return JSONResponse({"status": "ok", "observations": results})

@app.post("/reset/{task_id}")
def reset(task_id: str):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    obs = envs[task_id].reset()
    return obs.model_dump()


@app.post("/step/{task_id}")
def step(task_id: str, action: Action):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    env = envs[task_id]
    if env.df is None:
        raise HTTPException(status_code=400, detail="Call /reset/{task_id} first")
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state/{task_id}")
def state(task_id: str):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    env = envs[task_id]
    if env.df is None:
        raise HTTPException(status_code=400, detail="Call /reset/{task_id} first")
    return envs[task_id].state().model_dump()


# ------------------------------------------------------------------ #
#  list all tasks                                                      #
# ------------------------------------------------------------------ #
@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "task1", "name": "Patient Intake Null Imputation",   "difficulty": "easy"},
            {"id": "task2", "name": "E-Commerce Order Type Fixing",     "difficulty": "medium"},
            {"id": "task3", "name": "IoT Sensor Log Full Pipeline",     "difficulty": "hard"},
        ]
    }
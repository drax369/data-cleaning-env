import requests

BASE = "http://localhost:7860"

for task_id in ["task1", "task2", "task3"]:
    r = requests.post(f"{BASE}/reset/{task_id}")
    print(f"\n{task_id} reset status: {r.status_code}")

    action = {"action_type": "fill_null", "column": "age", "method": "median"}
    r2 = requests.post(f"{BASE}/step/{task_id}", json=action)
    print(f"{task_id} step status: {r2.status_code}")
    
    result = r2.json()
    reward = result.get("reward", {})
    print(f"{task_id} score={reward.get('score')} null={reward.get('null_score')} dtype={reward.get('dtype_score')} dupe={reward.get('duplicate_score')} outlier={reward.get('outlier_score')}")

    r3 = requests.get(f"{BASE}/state/{task_id}")
    state = r3.json()
    print(f"{task_id} state current_score: {state.get('current_score')}")
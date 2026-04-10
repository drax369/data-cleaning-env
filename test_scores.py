from environment.env import DataCleaningEnv
from environment.models import Action

env = DataCleaningEnv('task1')
obs = env.reset()
state = env.state()
print('current_score after reset:', state.current_score)

action = Action(action_type='fill_null', column='age', method='median')
obs, reward, done, info = env.step(action)
print('reward.score after step:', reward.score)
print('reward.null_score:', reward.null_score)
print('reward.dtype_score:', reward.dtype_score)
print('reward.duplicate_score:', reward.duplicate_score)
print('reward.outlier_score:', reward.outlier_score)
print('reward.efficiency_penalty:', reward.efficiency_penalty)
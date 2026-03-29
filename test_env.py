from environment.env import DataCleaningEnv
from environment.models import Action

env = DataCleaningEnv('task1')
obs = env.reset()
print('reset ok — nulls:', obs.null_counts)

action = Action(action_type='fill_null', column='age', method='median')
obs, reward, done, info = env.step(action)
print('after fill age — score:', reward.score)
print('action result:', info.get('action_result'))
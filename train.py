from stable_baselines3 import PPO
from utils.rl import SpaceInvadersEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(rank: int, seed: int = 0):
    def _init():
        env = SpaceInvadersEnv()
        return env
    return _init

env = SpaceInvadersEnv()

n_envs = 128

env = DummyVecEnv([make_env(i) for i in range(n_envs)])
model = PPO(
    "MlpPolicy", 
    env, 
    n_steps=8192,
    batch_size=256,
    n_epochs=7,
    learning_rate=0.001,
    verbose=1, 
    device="cpu", 
    gamma=0.985, 
    ent_coef=0.015,
    clip_range=0.2,
)
model.learn(75_000_000)
model.save("invader_agent")
from stable_baselines3 import PPO
from utils.ml import SpaceInvadersEnv

env = SpaceInvadersEnv()
model = PPO(
    "MlpPolicy", 
    env, 
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    verbose=1, 
    device="cpu", 
    gamma=0.99, 
    ent_coef=0.02,
    clip_range=0.2,
    gae_lambda=0.95
)
model.learn(1_000_000)
model.save("invader_agent")
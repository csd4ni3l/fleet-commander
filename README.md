Fleet Commander is like Space Invaders but you are the enemy instead of the player.

It uses AI (Reinforcement Learning) for the Player(s), and You, the Enemy has to defeat it.

I know the game is too easy and is too simple, but please understand that doing RL isn't the easiest thing ever. I also did this very late.

You can train it yourself, or use the default model(125 million timesteps, took 7 hours) which comes with the game.

# Install steps:

**It's important to install torch before stable_baselines3, otherwise it will use the GPU version, which i never tested with, and probably wouldn't be much faster because i depend on CPU for the simulation**

## For uv
- `uv sync`
- `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `uv pip install stable_baselines3`
- `uv run run.py`

## For pip
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip3 install -r requirements.txt`
- `pip3 install torch --index-url https://download.pytorch.org/whl/cpu`
- `pip3 install stable_baselines3`
- `python3 run.py`

# Disclaimer
AI assistance was used in this project, since i never did any RL work before. But every instance of AI code was heavily modified by me.
Fleet Commander is like Space Invaders but you are the enemy instead of the player.

It uses AI (Reinforcement Learning) for the Player, and you, the Enemy has to defeat it.

I know the game is too easy and is too simple, but please understand that doing RL isnt the easiest thing ever. I also did this so late so yeah.

You can train yourself, or use the default model(10 million timesteps) which comes with the game.

# Install steps:

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
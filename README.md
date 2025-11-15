Fleet Commander is like Space Invaders but you are the enemy instead of the player.

It uses AI (Reinforcement Learning) for the Player, and you, the Enemy has to defeat it.

You can train yourself, or use the default model which comes with the game.

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
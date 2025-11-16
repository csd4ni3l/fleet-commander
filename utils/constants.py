import arcade.color
from arcade.types import Color
from arcade.gui.widgets.buttons import UITextureButtonStyle, UIFlatButtonStyle
from arcade.gui.widgets.slider import UISliderStyle

ENEMY_SPEED = 5
ENEMY_ATTACK_SPEED = 0.75

PLAYER_SPEED = 5 # not actually player
PLAYER_ATTACK_SPEED = 0.75

BULLET_SPEED = 5
BULLET_RADIUS = 15

# default, min, max, step
MODEL_SETTINGS = {
    "n_steps": [1024, 256, 8192, 256],
    "batch_size": [128, 16, 512, 16],
    "n_epochs": [10, 1, 50, 1],
    "learning_rate": [3e-4, 1e-5, 1e-2, 1e-5],
    "gamma": [0.99, 0.8, 0.9999, 0.001],
    "ent_coef": [0.015, 0.0, 0.1, 0.001],
    "clip_range": [0.2, 0.1, 0.4, 0.01],
    "learning_steps": [1_000_000, 50_000, 25_000_000, 50_000],
    "n_envs": (12, 1, 128, 1)
}

DIFFICULTY_SETTINGS = {
    "enemy_rows": ["Enemy Rows", 1, 6],
    "enemy_cols": ["Enemy Columns", 1, 7],
    "enemy_respawns": ["Enemy Respawns", 1, 5],
    "player_count": ["Player Count", 1, 10],
    "player_respawns": ["Player Respawns", 1, 5]
}

DIFFICULTY_LEVELS = {
    "Easy": {
        "enemy_rows": 3,
        "enemy_cols": 4,
        "enemy_respawns": 5,
        "player_count": 2,
        "player_respawns": 2
    },
    "Medium": {
        "enemy_rows": 3,
        "enemy_cols": 5,
        "enemy_respawns": 4,
        "player_count": 4,
        "player_respawns": 3
    },
    "Hard": {
        "enemy_rows": 4,
        "enemy_cols": 6,
        "enemy_respawns": 3,
        "player_count": 6,
        "player_respawns": 4
    },
    "Extra Hard": {
        "enemy_rows": 6,
        "enemy_cols": 7,
        "enemy_respawns": 2,
        "player_count": 8,
        "player_respawns": 5
    },
    "Custom": {

    }
}

menu_background_color = (30, 30, 47)
log_dir = 'logs'
monitor_log_dir = "training_logs"
discord_presence_id = 1438214877343907881

button_style = {'normal': UITextureButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK), 'hover': UITextureButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK),
                'press': UITextureButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK), 'disabled': UITextureButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK)}
big_button_style = {'normal': UITextureButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK, font_size=26), 'hover': UITextureButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK, font_size=26),
                'press': UITextureButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK, font_size=26), 'disabled': UITextureButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK, font_size=26)}

dropdown_style = {'normal': UIFlatButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK, bg=Color(128, 128, 128)), 'hover': UIFlatButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK, bg=Color(49, 154, 54)),
                  'press': UIFlatButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK, bg=Color(128, 128, 128)), 'disabled': UIFlatButtonStyle(font_name="Roboto", font_color=arcade.color.BLACK, bg=Color(128, 128, 128))}

slider_default_style = UISliderStyle(bg=Color(128, 128, 128), unfilled_track=Color(128, 128, 128), filled_track=Color(49, 154, 54))
slider_hover_style = UISliderStyle(bg=Color(49, 154, 54), unfilled_track=Color(128, 128, 128), filled_track=Color(49, 154, 54))

slider_style = {'normal': slider_default_style, 'hover': slider_hover_style, 'press': slider_hover_style, 'disabled': slider_default_style}

settings = {
    "Graphics": {
        "Window Mode": {"type": "option", "options": ["Windowed", "Fullscreen", "Borderless"], "config_key": "window_mode", "default": "Windowed"},
        "Resolution": {"type": "option", "options": ["1366x768", "1440x900", "1600x900", "1920x1080", "2560x1440", "3840x2160"], "config_key": "resolution"},
        "Anti-Aliasing": {"type": "option", "options": ["None", "2x MSAA", "4x MSAA", "8x MSAA", "16x MSAA"], "config_key": "anti_aliasing", "default": "4x MSAA"},
        "VSync": {"type": "bool", "config_key": "vsync", "default": True},
        "FPS Limit": {"type": "slider", "min": 0, "max": 480, "config_key": "fps_limit", "default": 60},
    },
    "Sound": {
        "Music": {"type": "bool", "config_key": "music", "default": True},
        "SFX": {"type": "bool", "config_key": "sfx", "default": True},
        "Music Volume": {"type": "slider", "min": 0, "max": 100, "config_key": "music_volume", "default": 50},
        "SFX Volume": {"type": "slider", "min": 0, "max": 100, "config_key": "sfx_volume", "default": 50},
    },
    "Miscellaneous": {
        "Discord RPC": {"type": "bool", "config_key": "discord_rpc", "default": True},
    },
    "Credits": {}
}
settings_start_category = "Graphics"

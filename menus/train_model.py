import arcade, arcade.gui, threading, io, os, time

from utils.constants import button_style, MODEL_SETTINGS, monitor_log_dir
from utils.preload import button_texture, button_hovered_texture

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from io import BytesIO
from PIL import Image

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.logger import configure

from utils.rl import SpaceInvadersEnv

class TrainModel(arcade.gui.UIView):
    def __init__(self, pypresence_client):
        super().__init__()

        self.pypresence_client = pypresence_client
        self.pypresence_client.update(state="Model Training")

        self.anchor = self.add_widget(arcade.gui.UIAnchorLayout(size_hint=(1, 1)))
        self.box = self.anchor.add(arcade.gui.UIBoxLayout(space_between=10))

        self.settings = {
            setting: data[0] # default value
            for setting, data in MODEL_SETTINGS.items()
        }
        self.labels = {}

        self.training = False
        self.training_text = ""

        self.last_progress_update = time.perf_counter()

    def on_show_view(self):
        super().on_show_view()

        self.show_menu()

    def main_exit(self):
        from menus.main import Main
        self.window.show_view(Main(self.pypresence_client))

    def show_menu(self):
        self.back_button = self.anchor.add(arcade.gui.UITextureButton(texture=button_texture, texture_hovered=button_hovered_texture, text='<--', style=button_style, width=100, height=50), anchor_x="left", anchor_y="top", align_x=5, align_y=-5)
        self.back_button.on_click = lambda event: self.main_exit()

        self.box.add(arcade.gui.UILabel("Settings", font_size=36))

        for setting, data in MODEL_SETTINGS.items():
            default, min_value, max_value, step = data
            label = self.box.add(arcade.gui.UILabel(text=f"{setting.replace('_', ' ').capitalize()}: {default}", font_size=18))
            
            slider = self.box.add(arcade.gui.UISlider(value=default, min_value=min_value, max_value=max_value, step=step, width=self.window.width / 2, height=self.window.height / 25))
            slider._render_steps = lambda surface: None
            slider.on_change = lambda e, key=setting: self.change_value(key, e.new_value)

            self.labels[setting] = label

        train_button = self.box.add(arcade.gui.UITextureButton(width=self.window.width / 2, height=self.window.height / 10, text="Train", style=button_style, texture=button_texture, texture_hovered=button_hovered_texture))
        train_button.on_click = lambda e: self.start_training()

    def change_value(self, key, value):
        self.labels[key].text = f"{key.replace('_', ' ').capitalize()}: {self.round_near_int(value)}"
        self.settings[key] = self.round_near_int(value)

    def start_training(self):
        self.box.clear()

        self.training = True

        self.training_label = self.box.add(arcade.gui.UILabel("No Output yet.", font_size=16, multiline=True, width=self.window.width / 2, height=self.window.height / 2))

        self.plot_image_widget = self.box.add(arcade.gui.UIImage(texture=arcade.Texture.create_empty("empty", (1, 1))))
        self.plot_image_widget.visible = False

        threading.Thread(target=self.train, daemon=True).start()

    def on_update(self, delta_time):
        if self.training and os.path.exists(os.path.join("training_logs", "progress.csv")) and time.perf_counter() - self.last_progress_update >= 0.5:
            self.last_progress_update = time.perf_counter()
            
            try:
                progress_df = pd.read_csv(os.path.join("training_logs", "progress.csv"))
            except pd.errors.EmptyDataError:
                return
            
            progress_text = ""

            for key, value in progress_df.items():
                progress_text += f"{key}: {round(value.iloc[-1], 6)}\n"

            self.training_text = progress_text

        if hasattr(self, "training_label"):
            self.training_label.text = self.training_text

    def round_near_int(self, x, tol=1e-4):
        nearest = round(x)
        if abs(x - nearest) < tol:
            return nearest
        return x

    def train(self):
        os.makedirs(monitor_log_dir, exist_ok=True)
        env = Monitor(SpaceInvadersEnv(), filename=os.path.join(monitor_log_dir, "monitor.csv"))

        model = PPO(
            "MlpPolicy", 
            env, 
            n_steps=self.settings["n_steps"],
            batch_size=self.settings["batch_size"],
            n_epochs=self.settings["n_epochs"],
            learning_rate=self.settings["learning_rate"],
            verbose=1,
            device="cpu",
            gamma=self.settings["gamma"], 
            ent_coef=self.settings["ent_coef"],
            clip_range=self.settings["clip_range"],
        )

        new_logger = configure(
            folder=monitor_log_dir, format_strings=["csv"]
        )
        
        model.set_logger(new_logger)

        model.learn(self.settings["learning_steps"])
        model.save("invader_agent")

        self.training = False

        self.plot_results(os.path.join(monitor_log_dir, "monitor.csv"), os.path.join(monitor_log_dir, "progress.csv"))

    def plot_results(self, log_path, loss_log_path):
        df = pd.read_csv(log_path, skiprows=1)

        fig, axes = plt.subplots(2, 1, figsize=(6, 8), dpi=100)

        loss_df = pd.read_csv(loss_log_path)

        axes[0].plot(np.cumsum(df['l']), df['r'], label='Reward') 
        axes[0].set_title('PPO Training: Episodic Reward')
        axes[0].set_xlabel('Total Timesteps')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True)

        axes[1].plot(loss_df['time/total_timesteps'], loss_df['train/policy_gradient_loss'], label='Policy Gradient Loss')
        axes[1].plot(loss_df['time/total_timesteps'], loss_df['train/value_loss'], label='Value Loss')
        axes[1].plot(loss_df['time/total_timesteps'], loss_df['train/explained_variance'], label='Explained Variance')
        axes[1].set_title('PPO Training: Loss Functions')
        axes[1].set_xlabel('Total Timesteps')
        axes[1].set_ylabel('Loss Value')
        axes[1].legend()
        axes[1].grid(True)
    
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        plot_texture = arcade.Texture(Image.open(buffer))

        self.plot_image_widget.texture = plot_texture
        self.plot_image_widget.size_hint = (None, None)
        self.plot_image_widget.width = plot_texture.width
        self.plot_image_widget.height = plot_texture.height

        self.plot_image_widget.visible = True
        self.training_text = "Training finished. Plot displayed."
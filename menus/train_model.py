import arcade, arcade.gui, threading, os, queue, time, shutil

import matplotlib.pyplot as plt
import pandas as pd

from utils.constants import button_style, MODEL_SETTINGS, monitor_log_dir
from utils.preload import button_texture, button_hovered_texture
from utils.rl import SpaceInvadersEnv

from PIL import Image
from io import BytesIO

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(rank: int, seed: int = 0):
    def _init():
        env = SpaceInvadersEnv()
        env = Monitor(env, filename=os.path.join(monitor_log_dir, f"monitor_{rank}.csv"))
        return env
    return _init

class TrainModel(arcade.gui.UIView):
    def __init__(self, pypresence_client):
        super().__init__()

        self.pypresence_client = pypresence_client
        self.pypresence_client.update(state="Model Training")

        self.anchor = self.add_widget(arcade.gui.UIAnchorLayout(size_hint=(1, 1)))
        self.box = self.anchor.add(arcade.gui.UIBoxLayout(space_between=5))

        self.settings = {
            setting: data[0]
            for setting, data in MODEL_SETTINGS.items()
        }
        
        self.labels = {}

        self.training = False
        self.training_text = "Starting training..."
        
        self.result_queue = queue.Queue()
        self.training_thread = None

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

        self.box.add(arcade.gui.UILabel("Settings", font_size=32))

        for setting, data in MODEL_SETTINGS.items():
            default, min_value, max_value, step = data
            is_int = setting == "n_envs" or (abs(step - 1) < 1e-6 and abs(min_value - round(min_value)) < 1e-6)
            
            val_text = str(int(default)) if is_int else str(default)
            label = self.box.add(arcade.gui.UILabel(text=f"{setting.replace('_', ' ').capitalize()}: {val_text}", font_size=14))
            
            slider = self.box.add(arcade.gui.UISlider(value=default, min_value=min_value, max_value=max_value, step=step, width=self.window.width / 2, height=self.window.height / 25))
            slider._render_steps = lambda surface: None
            slider.on_change = lambda e, key=setting, is_int_slider=is_int: self.change_value(key, e.new_value, is_int_slider)

            self.labels[setting] = label

        train_button = self.box.add(arcade.gui.UITextureButton(width=self.window.width / 2, height=self.window.height / 10, text="Train", style=button_style, texture=button_texture, texture_hovered=button_hovered_texture))
        train_button.on_click = lambda e: self.start_training()

    def change_value(self, key, value, is_int=False):
        if is_int:
            val = int(round(value))
            self.settings[key] = val
            self.labels[key].text = f"{key.replace('_', ' ').capitalize()}: {val}"
        else:
            val = self.round_near_int(value)
            self.settings[key] = val
            self.labels[key].text = f"{key.replace('_', ' ').capitalize()}: {val}"

    def start_training(self):
        self.box.clear()

        self.training_text = "Starting training..."
        self.training_label = self.box.add(arcade.gui.UILabel("Starting training...", font_size=16, multiline=True, width=self.window.width / 2, height=self.window.height / 2))

        self.plot_image_widget = self.box.add(arcade.gui.UIImage(texture=arcade.Texture.create_empty("empty", (1, 1))))
        self.plot_image_widget.visible = False

        self.training_thread = threading.Thread(target=self.train, daemon=True)
        self.training_thread.start()

    def on_update(self, delta_time):
        try:
            result = self.result_queue.get_nowait()
            
            if result["type"] == "text":
                self.training_text = result["message"]
            
            elif result["type"] == "plot":
                self.plot_image_widget.texture = result["image"]
                self.plot_image_widget.width = result["image"].width
                self.plot_image_widget.height = result["image"].height
                self.plot_image_widget.trigger_render()
                self.plot_image_widget.visible = True

            elif result["type"] == "finished":
                self.training = False
                self.training_text = "Training finished."

        except queue.Empty:
            if self.training and os.path.exists(os.path.join("training_logs", "progress.csv")) and all([os.path.exists(os.path.join(monitor_log_dir, f"monitor_{i}.csv.monitor.csv")) for i in range(int(self.settings["n_envs"]))]) and time.perf_counter() - self.last_progress_update >= 1:
                self.last_progress_update = time.perf_counter()
                self.plot_results()

        if hasattr(self, "training_label"):
            self.training_label.text = self.training_text

    def round_near_int(self, x, tol=1e-4):
        nearest = round(x)
        
        if abs(x - nearest) < tol:
            return nearest
        
        return x

    def train(self):
        if os.path.exists(monitor_log_dir):
            shutil.rmtree(monitor_log_dir)
        os.makedirs(monitor_log_dir)

        n_envs = int(self.settings["n_envs"])
        env = DummyVecEnv([make_env(i) for i in range(n_envs)])

        n_steps = int(self.settings["n_steps"])
        batch_size = int(self.settings["batch_size"])

        total_steps_per_rollout = n_steps * max(1, n_envs)
        if total_steps_per_rollout % batch_size != 0:
            batch_size = max(64, total_steps_per_rollout // max(1, total_steps_per_rollout // batch_size))
            print(f"Warning: Adjusting batch size to {batch_size} for {n_envs} envs.")

        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=int(self.settings["n_epochs"]),
            learning_rate=float(self.settings["learning_rate"]),
            verbose=1,
            device="cpu",
            gamma=float(self.settings["gamma"]),
            ent_coef=float(self.settings["ent_coef"]),
            clip_range=float(self.settings["clip_range"]),
        )

        new_logger = configure(folder=monitor_log_dir, format_strings=["csv"])
        model.set_logger(new_logger)

        try:
            self.training = True
            model.learn(int(self.settings["learning_steps"]))
            model.save("invader_agent")
        except Exception as e:
            print(f"Error during training: {e}")
            self.result_queue.put({"type": "text", "message": f"Error:\n{e}"})
        finally:
            try:
                env.close()
            except Exception:
                pass
            self.result_queue.put({"type": "finished"})

    def plot_results(self):
        try:
            reward_df = pd.read_csv(os.path.join(monitor_log_dir, "progress.csv"))
        except pd.errors.EmptyDataError:
            return

        all_monitor_files = [os.path.join(monitor_log_dir, f) for f in os.listdir(monitor_log_dir) if f.startswith("monitor_") and f.endswith(".csv")]
        try:
            df_list = [pd.read_csv(f, skiprows=1) for f in all_monitor_files]
        except pd.errors.EmptyDataError:
            return

        monitor_df = pd.concat(df_list).sort_values(by='t')
        monitor_df['total_timesteps'] = monitor_df['l'].cumsum()

        loss_log_path = os.path.join(monitor_log_dir, "progress.csv")
        loss_df = None
        if os.path.exists(loss_log_path):
            try:
                loss_df = pd.read_csv(loss_log_path)
            except Exception:
                loss_df = None

        fig, axes = plt.subplots(2, 1, figsize=(6, 8), dpi=100)

        if monitor_df is not None and 'total_timesteps' in monitor_df.columns and 'r' in monitor_df.columns:
            axes[0].plot(monitor_df['total_timesteps'], monitor_df['r'].rolling(window=10).mean(), label='Episodic Reward (Rolling 10)')
        elif reward_df is not None and 'time/total_timesteps' in reward_df.columns and 'rollout/ep_rew_mean' in reward_df.columns:
            axes[0].plot(reward_df['time/total_timesteps'], reward_df['rollout/ep_rew_mean'], label='Ep reward mean')
        else:
            axes[0].text(0.5, 0.5, "No reward data available", horizontalalignment='center', verticalalignment='center')
        
        axes[0].set_title('PPO Training: Episodic Reward')
        axes[0].set_xlabel('Total Timesteps')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True)

        axes[1].set_title('PPO Training: Loss & Variance')
        axes[1].set_xlabel('Total Timesteps')
        axes[1].set_ylabel('Value')
        axes[1].grid(True)

        if loss_df is not None and 'time/total_timesteps' in loss_df.columns and 'train/policy_gradient_loss' in loss_df.columns and 'train/value_loss' in loss_df.columns and 'train/explained_variance' in loss_df.columns:
            tcol = 'time/total_timesteps'
            axes[1].plot(loss_df[tcol], loss_df['train/policy_gradient_loss'], label='Policy Loss')
            axes[1].plot(loss_df[tcol], loss_df['train/value_loss'], label='Value Loss')
            axes[1].plot(loss_df[tcol], loss_df['train/explained_variance'], label='Explained Variance')

            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "No loss/variance data available", horizontalalignment='center', verticalalignment='center')

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        pil_img = Image.open(buffer).convert("RGBA")

        plot_texture = arcade.Texture(pil_img)

        self.result_queue.put({"type": "plot", "image": plot_texture})

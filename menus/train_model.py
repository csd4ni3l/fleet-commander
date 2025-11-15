import arcade, arcade.gui

from utils.constants import button_style, MODEL_SETTINGS
from utils.preload import button_texture, button_hovered_texture

from stable_baselines3 import PPO
from utils.ml import SpaceInvadersEnv

class TrainModel(arcade.gui.UIView):
    def __init__(self, pypresence_client):
        self.pypresence_client = pypresence_client
        self.pypresence_client.update(state="Model Training")

        self.current_state = "settings"

        self.anchor = self.add_widget(arcade.gui.UIAnchorLayout(size_hint=(1, 1)))
        self.box = self.anchor.add(arcade.gui.UIBoxLayout(space_between=10))

        self.settings = MODEL_SETTINGS.copy()

    def on_show_view(self):
        super().on_show_view()

        self.show_menu(self.current_state)

    def show_menu(self, state):
        if state == "settings":
            self.box.add(arcade.gui.UILabel("Settings", font_size=48))

            for setting, data in MODEL_SETTINGS:
                default, min_value, max_value, step = data 
                self.box.add(arcade.gui.UILabel(text=f"{setting.replace('_', ' ').capitalize()}: {default}"))
                
                slider = self.box.add(arcade.gui.UISlider(value=default, min_value=min_value, max_value=max_value, step=step))
                slider._render_steps = lambda surface: None
                slider.on_change = lambda e, key=setting: self.change_value(key, e.new_value)

            train_button = self.box.add(arcade.gui.UITextureButton(width=self.window.width / 2, height=self.window.height / 10, text="Train", style=button_style, texture=button_texture, texture_hovered=button_hovered_texture))
            train_button.on_click = lambda e: self.train()

    def change_value(self, key, value):
        ...
        
    def train(self):
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
            ent_coef=0.01,
            clip_range=0.2
        )
        model.learn(1_000_000)
        model.save("invader_agent")
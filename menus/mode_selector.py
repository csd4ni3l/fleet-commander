import arcade, arcade.gui

from utils.preload import button_texture, button_hovered_texture
from utils.constants import dropdown_style, button_style, DIFFICULTY_LEVELS, DIFFICULTY_SETTINGS

class ModeSelector(arcade.gui.UIView):
    def __init__(self, pypresence_client):
        super().__init__()

        self.pypresence_client = pypresence_client
        self.pypresence_client.update(state="Selecting Mode")

        self.anchor = self.add_widget(arcade.gui.UIAnchorLayout(size_hint=(1, 1)))
        self.box = self.anchor.add(arcade.gui.UIBoxLayout(size_hint=(0.75, 0.75), space_between=10), anchor_x="center", anchor_y="center")

        self.settings = DIFFICULTY_LEVELS["Easy"]
        self.setting_sliders = {}
        self.setting_labels = {}

    def on_show_view(self):
        super().on_show_view()

        self.back_button = self.anchor.add(arcade.gui.UITextureButton(texture=button_texture, texture_hovered=button_hovered_texture, text='<--', style=button_style, width=100, height=50), anchor_x="left", anchor_y="top", align_x=5, align_y=-5)
        self.back_button.on_click = lambda event: self.main_exit()

        self.box.add(arcade.gui.UILabel("Settings", font_size=32))

        self.box.add(arcade.gui.UISpace(height=self.window.height / 80))

        self.difficulty_selector = self.box.add(arcade.gui.UIDropdown(default="Easy", options=list(DIFFICULTY_LEVELS.keys()), active_style=dropdown_style, primary_style=dropdown_style, dropdown_style=dropdown_style, width=self.window.width / 2, height=self.window.height / 20))
        self.difficulty_selector.on_change = lambda event: self.set_difficulty_values(event.new_value)

        self.box.add(arcade.gui.UISpace(height=self.window.height / 80))

        for key, data in DIFFICULTY_SETTINGS.items():
            default, name, min_value, max_value = DIFFICULTY_LEVELS["Easy"][key], *data
            
            label = self.box.add(arcade.gui.UILabel(text=f"{name}: {default}", font_size=14))

            slider = self.box.add(arcade.gui.UISlider(value=default, min_value=min_value, max_value=max_value, step=1, width=self.window.width / 2, height=self.window.height / 25))
            slider._render_steps = lambda surface: None
            slider.on_event = lambda event: None # disable slider for difficulties
            slider.on_click = lambda event: None # disable slider for difficulties
            slider.on_change = lambda e, key=key: self.change_value(key, e.new_value)

            self.setting_sliders[key] = slider
            self.setting_labels[key] = label

        self.play_button = self.box.add(arcade.gui.UITextureButton(text="Play", width=self.window.width / 2, height=self.window.height / 15, texture=button_texture, texture_hovered=button_hovered_texture, style=button_style))
        self.play_button.on_click = lambda event: self.start_game()

    def set_difficulty_values(self, difficulty):
        for key, value in DIFFICULTY_LEVELS[difficulty].items():
            self.settings[key] = value
            self.setting_labels[key].text = f"{DIFFICULTY_SETTINGS[key][0]}: {value}"
            self.setting_sliders[key].value = value

        for slider in self.setting_sliders.values():
            if difficulty != "Custom":
                slider.on_event = lambda event: None
                slider.on_click = lambda event: None
            else:
                slider.on_event = lambda event, slider=slider: arcade.gui.UISlider.on_event(slider, event)
                slider.on_click = lambda event, slider=slider: arcade.gui.UISlider.on_click(slider, event)

    def change_value(self, key, value):
        self.settings[key] = int(value)
        self.setting_labels[key].text = f"{DIFFICULTY_SETTINGS[key][0]}: {int(value)}"

    def start_game(self):
        from game.play import Game
        self.window.show_view(Game(self.pypresence_client, self.settings))
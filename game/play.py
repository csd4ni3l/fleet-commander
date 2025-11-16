import arcade, arcade.gui, random, time

from utils.constants import button_style, ENEMY_ATTACK_SPEED, ENEMY_SPEED
from utils.preload import button_texture, button_hovered_texture

from stable_baselines3 import PPO

from game.sprites import EnemyFormation, Player, Bullet

class Game(arcade.gui.UIView):
    def __init__(self, pypresence_client, settings):
        super().__init__()

        self.settings = settings
        self.pypresence_client = pypresence_client
        self.pypresence_client.update(state="Invading Space")

        self.anchor = self.add_widget(arcade.gui.UIAnchorLayout(size_hint=(1, 1)))
        
        self.spritelist = arcade.SpriteList()

        self.players = []
        for _ in range(settings["player_count"]):
            self.players.append(Player(self.window.width / 2 + random.randint(int(-self.window.width / 3), int(self.window.width / 3)), 100))  # not actually player
            self.spritelist.append(self.players[-1])
        
        self.model = PPO.load("invader_agent.zip")

        self.enemy_formation = EnemyFormation(self.window.width / 2 + random.randint(int(-self.window.width / 3), int(self.window.width / 3)), self.window.height * 0.9, self.spritelist, settings["enemy_rows"], settings["enemy_cols"])
        self.player_bullets: list[Bullet] = []
        self.enemy_bullets: list[Bullet] = []

        self.player_respawns = settings["player_respawns"]
        self.enemy_respawns = settings["enemy_respawns"]

        self.score = 0

        self.last_enemy_shoot = time.perf_counter()

        self.game_over = False

    def on_show_view(self):
        super().on_show_view()

        self.back_button = self.anchor.add(arcade.gui.UITextureButton(texture=button_texture, texture_hovered=button_hovered_texture, text='<--', style=button_style, width=100, height=50), anchor_x="left", anchor_y="top", align_x=5, align_y=-5)
        self.back_button.on_click = lambda event: self.main_exit()
        self.score_label = self.anchor.add(arcade.gui.UILabel("Score: 0", font_size=24), anchor_x="center", anchor_y="top")

    def main_exit(self):
        from menus.main import Main
        self.window.show_view(Main(self.pypresence_client))

    def on_update(self, delta_time):
        if self.game_over:
            return

        if self.window.keyboard[arcade.key.LEFT] or self.window.keyboard[arcade.key.A]:
            self.enemy_formation.move(self.window.width, self.window.height, "x", -ENEMY_SPEED)
        if self.window.keyboard[arcade.key.RIGHT] or self.window.keyboard[arcade.key.D]:
            self.enemy_formation.move(self.window.width, self.window.height, "x", ENEMY_SPEED)
        if self.window.keyboard[arcade.key.DOWN] or self.window.keyboard[arcade.key.S]:
            self.enemy_formation.move(self.window.width, self.window.height, "y", -ENEMY_SPEED)
        if self.window.keyboard[arcade.key.UP] or self.window.keyboard[arcade.key.W]:
            self.enemy_formation.move(self.window.width, self.window.height, "y", ENEMY_SPEED)
        if self.enemy_formation.enemies and self.window.keyboard[arcade.key.SPACE] and time.perf_counter() - self.last_enemy_shoot >= ENEMY_ATTACK_SPEED:
            self.last_enemy_shoot = time.perf_counter()
            enemy = self.enemy_formation.get_lowest_enemy()
            self.shoot(enemy.center_x, enemy.center_y, -1)

        bullets_to_remove = []

        for bullet in self.player_bullets + self.enemy_bullets:
            bullet.update()
            
            bullet_hit = False
            if bullet.direction_y == 1:
                for enemy in self.enemy_formation.enemies:
                    if bullet.rect.intersection(enemy.rect):
                        self.enemy_formation.remove_enemy(enemy)
                        bullets_to_remove.append(bullet)
                        bullet_hit = True
                        break
            else:
                for player in self.players:
                    if bullet.rect.intersection(player.rect):
                        self.spritelist.remove(player)
                        self.players.remove(player)
                        bullets_to_remove.append(bullet)
                        bullet_hit = True
                        self.score += 75
                        break
                    
            if not bullet_hit and bullet.center_y > self.window.height or bullet.center_y < 0:
                bullets_to_remove.append(bullet)

        for bullet_to_remove in bullets_to_remove:
            self.spritelist.remove(bullet_to_remove)

            if bullet_to_remove in self.enemy_bullets:
                self.enemy_bullets.remove(bullet_to_remove)
            elif bullet_to_remove in self.player_bullets:
                self.player_bullets.remove(bullet_to_remove)

        for player in self.players:
            player.update(self.model, self.enemy_formation, self.enemy_bullets, self.window.width, self.window.height, self.player_respawns / self.settings["player_respawns"], self.enemy_respawns / self.settings["enemy_respawns"]) # not actually player

            if player.center_x > self.window.width:
                player.center_x = self.window.width
            elif player.center_x < 0:
                player.center_x = 0

            if player.shoot:
                player.shoot = False
                self.shoot(player.center_x, player.center_y, 1)

        if len(self.players) == 0:
            if self.player_respawns > 0:
                self.player_respawns -= 1
                for _ in range(self.settings["player_count"]):
                    self.players.append(Player(self.window.width / 2 + random.randint(int(-self.window.width / 3), int(self.window.width / 3)), 100))  # not actually player
                    self.spritelist.append(self.players[-1])
                self.score += 300
            else:
                self.game_over = True
                self.game_over_label = self.anchor.add(arcade.gui.UILabel("You (The enemies) won!", font_size=48), anchor_x="center", anchor_y="center")

        elif len(self.enemy_formation.enemies) == 0:
            if self.enemy_respawns > 0:
                self.enemy_respawns -= 1
                self.enemy_formation.create_formation(self.window.width / 2 + random.randint(int(-self.window.width / 3), int(self.window.width / 3)), self.window.height * 0.9)
            else:
                self.game_over = True
                self.game_over_label = self.anchor.add(arcade.gui.UILabel("You lost! The Players win!", font_size=48), anchor_x="center", anchor_y="center")

        self.score += 5 * delta_time
        
        self.score_label.text = f"Score: {int(self.score)}"

    def shoot(self, x, y, direction_y):
        bullet = Bullet(x, y, direction_y)
        self.spritelist.append(bullet)

        if direction_y == 1:
            bullets = self.player_bullets
        else:
            bullets = self.enemy_bullets

        bullets.append(bullet)

    def on_draw(self):
        super().on_draw()

        self.spritelist.draw()
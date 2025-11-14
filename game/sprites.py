import arcade, random, time

from utils.constants import PLAYER_SPEED, BULLET_SPEED, BULLET_RADIUS
from utils.preload import player_texture, enemy_texture

class Bullet(arcade.Sprite):
    def __init__(self, x, y, direction_y):
        super().__init__(arcade.texture.make_circle_texture(BULLET_RADIUS, arcade.color.YELLOW), center_x=x, center_y=y)

        self.direction_y = direction_y

    def update(self):
        self.center_y += self.direction_y * BULLET_SPEED

class Enemy(arcade.Sprite):
    def __init__(self, x, y):
        super().__init__(enemy_texture, center_x=x, center_y=y)

class Player(arcade.Sprite): # Not actually the player
    def __init__(self, x, y):
        super().__init__(player_texture, center_x=x, center_y=y)

        self.last_target_change = time.perf_counter()
        self.target = None

    def update(self, enemies):
        if not enemies:
            return

        if not self.target or time.perf_counter() - self.last_target_change >= 1:
            self.last_target_change = time.perf_counter()
            self.target = random.choice(enemies)

        if self.target.center_x > self.center_x:
            self.center_x += PLAYER_SPEED
        elif self.target.center_x < self.center_x:
            self.center_x -= PLAYER_SPEED 
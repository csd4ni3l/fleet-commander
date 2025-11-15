import arcade, time

from stable_baselines3 import PPO

import numpy as np

from utils.constants import PLAYER_SPEED, BULLET_SPEED, BULLET_RADIUS, PLAYER_ATTACK_SPEED, ENEMY_COLS, ENEMY_ROWS
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
        self.last_shoot = time.perf_counter()
        self.target = None
        self.shoot = False

        self.player_speed = 0

    def update(self, model: PPO, enemies, bullets, width, height):
        if enemies:
            nearest_enemy = min(enemies, key=lambda e: abs(e.center_x - self.center_x))
            enemy_x = (nearest_enemy.center_x - self.center_x) / width
            enemy_y = (nearest_enemy.center_y - self.center_y) / height
        else:
            enemy_x = 2
            enemy_y = 2

        enemy_count = len(enemies) / float(max(1, ENEMY_ROWS * ENEMY_COLS))
        player_x_norm = self.center_x / width

        curr_bullet = min(bullets, key=lambda b: abs(b.center_x - self.center_x) + abs(b.center_y - self.center_y)) if bullets else None
        if curr_bullet is not None:
            curr_bx = (curr_bullet.center_x - self.center_x) / float(width)
            curr_by = (curr_bullet.center_y - self.center_y) / float(height)
        else:
            curr_bx = 2.0
            curr_by = 2.0

        lowest = max(enemies, key=lambda e: e.center_y) if enemies else None
        if lowest is not None:
            lowest_dy = (lowest.center_y - self.center_y) / float(height)
        else:
            lowest_dy = 2.0

        enemy_dispersion = 0.0
        if enemies:
            xs = np.array([e.center_x for e in enemies], dtype=np.float32)
            enemy_dispersion = float(xs.std()) / float(width)

        obs = np.array([player_x_norm, enemy_x, enemy_y, lowest_dy, curr_bx, curr_by, self.player_speed, enemy_count, enemy_dispersion], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)

        self.prev_x = self.center_x
        if action == 0:
            self.center_x -= PLAYER_SPEED
        elif action == 1:
            self.center_x += PLAYER_SPEED
        elif action == 2:
            t = time.perf_counter()
            if t - self.last_shoot >= PLAYER_ATTACK_SPEED:
                self.last_shoot = t
                self.shoot = True

        self.player_speed = (self.center_x - self.prev_x) / max(1e-6, PLAYER_SPEED)

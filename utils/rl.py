import gymnasium as gym
import numpy as np
import arcade
import time
import random

from game.sprites import Enemy, Player, Bullet
from utils.constants import PLAYER_SPEED, BULLET_SPEED, PLAYER_ATTACK_SPEED, ENEMY_ROWS, ENEMY_COLS

class SpaceInvadersEnv(gym.Env):
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(9,), dtype=np.float32)

        self.enemies = []
        self.bullets = []
        self.dir_history = []
        self.last_shot = 0.0
        self.player = None
        self.prev_x = 0.0
        self.player_speed = 0.0
        self.prev_bx = 2.0
        self.steps_since_direction_change = 0
        self.last_direction = 0
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.enemies = []
        self.bullets = []
        self.dir_history = []
        self.player = Player(self.width / 2 + random.randint(int(-self.width / 3), int(self.width / 3)), 100)
        self.prev_x = self.player.center_x
        self.player_speed = 0.0
        self.prev_bx = 2.0
        self.steps_since_direction_change = 0
        self.last_direction = 0
        self.current_step = 0

        start_x = self.width * 0.15
        start_y = self.height * 0.9
        
        for r in range(ENEMY_ROWS):
            for c in range(ENEMY_COLS):
                e = Enemy(start_x + c * 100, start_y - r * 100)
                self.enemies.append(e)
        
        self.last_shot = time.perf_counter()
        return self._obs(), {}

    def _nearest_enemy(self):
        if not self.enemies:
            return None
        return min(self.enemies, key=lambda e: abs(e.center_x - self.player.center_x))

    def _lowest_enemy(self):
        if not self.enemies:
            return None
        return max(self.enemies, key=lambda e: e.center_y)

    def _nearest_enemy_bullet(self):
        enemy_bullets = [b for b in self.bullets if b.direction_y == -1]
        if not enemy_bullets:
            return None
        return min(enemy_bullets, key=lambda b: abs(b.center_x - self.player.center_x) + abs(b.center_y - self.player.center_y))

    def _obs(self):
        if self.enemies:
            nearest = self._nearest_enemy()
            enemy_x = (nearest.center_x - self.player.center_x) / float(self.width)
            enemy_y = (nearest.center_y - self.player.center_y) / float(self.height)
        else:
            enemy_x = 2.0
            enemy_y = 2.0

        lowest = self._lowest_enemy()

        if lowest is not None:
            lowest_dy = (lowest.center_y - self.player.center_y) / float(self.height)
        else:
            lowest_dy = 2.0

        nb = self._nearest_enemy_bullet()
        if nb is not None:
            bx = (nb.center_x - self.player.center_x) / float(self.width)
            by = (nb.center_y - self.player.center_y) / float(self.height)
        else:
            bx = 2.0
            by = 2.0

        enemy_count = len(self.enemies) / float(max(1, ENEMY_ROWS * ENEMY_COLS))
        player_x_norm = self.player.center_x / float(self.width)
        enemy_dispersion = 0.0
        
        if self.enemies:
            xs = np.array([e.center_x for e in self.enemies], dtype=np.float32)
            enemy_dispersion = float(xs.std()) / float(self.width)
        
        obs = np.array([player_x_norm, enemy_x, enemy_y, lowest_dy, bx, by, self.player_speed, enemy_count, enemy_dispersion], dtype=np.float32)
        return obs

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        nearest = self._nearest_enemy()
        if nearest is not None:
            enemy_x = (nearest.center_x - self.player.center_x) / float(self.width)
        else:
            enemy_x = 2.0

        prev_x = self.player.center_x
        current_action_dir = 0

        if action == 0:
            self.player.center_x -= PLAYER_SPEED
            current_action_dir = -1
        elif action == 1:
            self.player.center_x += PLAYER_SPEED
            current_action_dir = 1
        elif action == 2:
            t = time.perf_counter()
            if t - self.last_shot >= PLAYER_ATTACK_SPEED:
                self.last_shot = t
                
                b = Bullet(self.player.center_x, self.player.center_y, 1)

                self.bullets.append(b)
                
                if enemy_x != 2.0 and abs(enemy_x) < 0.04:
                    reward += 0.3
                elif enemy_x != 2.0 and abs(enemy_x) < 0.1:
                    reward += 0.1

        if self.player.center_x > self.width:
            self.player.center_x = self.width
        elif self.player.center_x < 0:
            self.player.center_x = 0

        self.player_speed = (self.player.center_x - prev_x) / max(1e-6, PLAYER_SPEED)

        if current_action_dir != 0:
            if self.last_direction != 0 and current_action_dir != self.last_direction:
                if self.steps_since_direction_change < 3:
                    reward -= 0.1

                self.steps_since_direction_change = 0
            else:
                self.steps_since_direction_change += 1
            self.last_direction = current_action_dir

        if enemy_x != 2.0:
            if abs(enemy_x) < 0.03:
                reward += 0.1
            elif abs(enemy_x) < 0.08:
                reward += 0.05

        for b in list(self.bullets):
            b.center_y += b.direction_y * BULLET_SPEED
            if b.center_y > self.height or b.center_y < 0:
                try:
                    self.bullets.remove(b)
                except ValueError:
                    pass

        for b in list(self.bullets):
            if b.direction_y == 1:
                for e in list(self.enemies):
                    if arcade.check_for_collision(b, e):
                        try:
                            self.enemies.remove(e)
                        except ValueError:
                            pass
                        try:
                            self.bullets.remove(b)
                        except ValueError:
                            pass
                        reward += 1.0
                        break

        for b in list(self.bullets):
            if b.direction_y == -1:
                if arcade.check_for_collision(b, self.player):
                    try:
                        self.bullets.remove(b)
                    except ValueError:
                        pass
                    reward -= 5.0
                    terminated = True

        if not self.enemies:
            reward += 10.0
            terminated = True

        if self.enemies and random.random() < 0.05:
            e = random.choice(self.enemies)
            b = Bullet(e.center_x, e.center_y, -1)
            self.bullets.append(b)

        curr_bullet = self._nearest_enemy_bullet()
        if curr_bullet is not None:
            curr_bx = (curr_bullet.center_x - self.player.center_x) / float(self.width)
        else:
            curr_bx = 2.0

        if self.prev_bx != 2.0 and curr_bx != 2.0:
            if abs(curr_bx) > abs(self.prev_bx):
                reward += 0.02

        reward -= 0.01

        obs = self._obs()
        self.prev_bx = curr_bx

        return obs, float(reward), bool(terminated), bool(truncated), {}
import gymnasium as gym
import numpy as np
import arcade
import random

from game.sprites import EnemyFormation, Player, Bullet
from utils.constants import PLAYER_SPEED, BULLET_SPEED, ENEMY_SPEED, DIFFICULTY_LEVELS

class SpaceInvadersEnv(gym.Env):
    def __init__(self, width=800, height=600, difficulty="Hard"):
        self.width = width
        self.height = height
        
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(12,), dtype=np.float32)

        if difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(f"Unknown difficulty: {difficulty}. Available: {list(DIFFICULTY_LEVELS.keys())}")
            
        self.difficulty_settings = DIFFICULTY_LEVELS[difficulty]

        self.bullets = []
        self.player = None
        self.enemy_formation = None
        self.player_speed = 0.0
        self.max_steps = 2000 
        self.current_step = 0
        self.enemies_killed = 0
        self.enemy_move_speed = ENEMY_SPEED
        self.player_respawns = self.difficulty_settings["player_respawns"]
        self.enemy_respawns = self.difficulty_settings["enemy_respawns"]
        self.player_respawns_remaining = 0
        self.enemy_respawns_remaining = 0
        self.player_alive = True
        
        self.player_attack_cooldown_steps = 5 
        self.current_cooldown = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.bullets = []
        self.player = Player(self.width / 2 + random.randint(int(-self.width / 3), int(self.width / 3)), 100)
        self.player_speed = 0.0
        self.current_step = 0
        self.enemies_killed = 0
        self.player_respawns_remaining = self.player_respawns
        self.enemy_respawns_remaining = self.enemy_respawns
        self.player_alive = True
        self.current_cooldown = 0
        
        start_x = self.width * 0.15
        start_y = self.height * 0.9
        
        self.enemy_formation = EnemyFormation(start_x, start_y, None, 
                                              self.difficulty_settings["enemy_rows"], 
                                              self.difficulty_settings["enemy_cols"])
        
        return self._obs(), {}

    def _nearest_enemy(self):
        if not self.enemy_formation.enemies:
            return None
        
        return min(self.enemy_formation.enemies, key=lambda e: abs(e.center_x - self.player.center_x))

    def _lowest_enemy(self):
        if not self.enemy_formation.enemies:
            return None
        
        return min(self.enemy_formation.enemies, key=lambda e: e.center_y)

    def _nearest_enemy_bullet(self):
        enemy_bullets = [b for b in self.bullets if b.direction_y == -1]
        
        if not enemy_bullets:
            return None
        
        return min(enemy_bullets, key=lambda b: abs(b.center_x - self.player.center_x) + abs(b.center_y - self.player.center_y))

    def _obs(self):
        if self.enemy_formation.enemies and self.player_alive:
            nearest = self._nearest_enemy()
            enemy_x = (nearest.center_x - self.player.center_x) / float(self.width)
            enemy_y = (nearest.center_y - self.player.center_y) / float(self.height)
        else:
            enemy_x = 2.0
            enemy_y = 2.0

        lowest = self._lowest_enemy()
        if lowest is not None and self.player_alive:
            lowest_dy = (lowest.center_y - self.player.center_y) / float(self.height)
        else:
            lowest_dy = 2.0

        nb = self._nearest_enemy_bullet()
        if nb is not None and self.player_alive:
            bx = (nb.center_x - self.player.center_x) / float(self.width)
            by = (nb.center_y - self.player.center_y) / float(self.height)
        else:
            bx = 2.0
            by = 2.0

        enemy_count = len(self.enemy_formation.enemies) / float(max(1, self.difficulty_settings["enemy_rows"] * self.difficulty_settings["enemy_cols"]))
        player_x_norm = self.player.center_x / float(self.width) if self.player_alive else 0.5
        
        enemy_dispersion = 0.0
        if self.enemy_formation.enemies:
            xs = np.array([e.center_x for e in self.enemy_formation.enemies], dtype=np.float32)
            enemy_dispersion = float(xs.std()) / float(self.width)
        
        can_shoot = 1.0 if (self.player_alive and self.current_cooldown <= 0) else 0.0
        
        player_respawns_norm = self.player_respawns_remaining / float(max(1, self.player_respawns))
        enemy_respawns_norm = self.enemy_respawns_remaining / float(max(1, self.enemy_respawns))

        obs = np.array([
            player_x_norm, 
            enemy_x, 
            enemy_y, 
            lowest_dy, 
            bx, 
            by, 
            self.player_speed, 
            enemy_count, 
            enemy_dispersion,
            can_shoot,
            player_respawns_norm,
            enemy_respawns_norm
        ], dtype=np.float32)
        
        return obs

    def _respawn_player(self):
        self.player = Player(self.width / 2 + random.randint(int(-self.width / 3), int(self.width / 3)), 100)
        self.player_alive = True
        self.bullets = [b for b in self.bullets if b.direction_y == 1]
        self.current_cooldown = 0

    def _respawn_enemies(self):
        self.enemy_formation.start_x = self.width * 0.15
        self.enemy_formation.start_y = self.height * 0.9
        self.enemy_formation.create_formation()

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        if self.current_cooldown > 0:
            self.current_cooldown -= 1

        if self.player_alive:
            prev_x = self.player.center_x

            if action == 0:
                self.player.center_x -= PLAYER_SPEED
            elif action == 1:
                self.player.center_x += PLAYER_SPEED
            elif action == 2:
                pass
            elif action == 3:
                if self.current_cooldown <= 0:
                    self.current_cooldown = self.player_attack_cooldown_steps
                    reward += 0.01
                    b = Bullet(self.player.center_x, self.player.center_y, 1)
                    self.bullets.append(b)
                else:
                    reward -= 0.02

            if self.enemy_formation.enemies:
                nearest = self._nearest_enemy()
                alignment = abs(nearest.center_x - self.player.center_x) / self.width
                if alignment < 0.025:
                    reward += 0.005

            self.player.center_x = np.clip(self.player.center_x, 0, self.width)
            self.player_speed = (self.player.center_x - prev_x) / max(1e-6, PLAYER_SPEED)

        if self.enemy_formation.enemies and self.player_alive:
            if self.enemy_formation.center_x < self.player.center_x:
                self.enemy_formation.move(self.width, self.height, "x", self.enemy_move_speed)
            elif self.enemy_formation.center_x > self.player.center_x:
                self.enemy_formation.move(self.width, self.height, "x", -self.enemy_move_speed)
            
            if random.random() < 0.02:
                if random.random() < 0.5:
                    self.enemy_formation.move(self.width, self.height, "y", -self.enemy_move_speed)
                else:
                    self.enemy_formation.move(self.width, self.height, "y", self.enemy_move_speed)

        bullets_to_remove = []

        for b in self.bullets:
            b.center_y += b.direction_y * BULLET_SPEED
            
            if b.center_y > self.height or b.center_y < 0:
                bullets_to_remove.append(b)
                continue
            
            if b.direction_y == 1:
                for e in self.enemy_formation.enemies:
                    if arcade.check_for_collision(b, e):
                        self.enemy_formation.remove_enemy(e)
                        bullets_to_remove.append(b)
                        reward += 10.0
                        self.enemies_killed += 1
                        break
            
            elif b.direction_y == -1 and self.player_alive:
                if arcade.check_for_collision(b, self.player):
                    bullets_to_remove.append(b)
                    reward -= 10.0
                    self.player_alive = False
                    
                    if self.player_respawns_remaining > 0:
                        self.player_respawns_remaining -= 1
                        self._respawn_player()
                    else:
                        terminated = True

        for b in bullets_to_remove:
            if b in self.bullets:
                self.bullets.remove(b)
        
        if self.player_alive:
            lowest_enemy = self._lowest_enemy()
            if lowest_enemy and lowest_enemy.center_y <= self.player.center_y:
                reward -= 10.0
                self.player_alive = False
                
                if self.player_respawns_remaining > 0:
                    self.player_respawns_remaining -= 1
                    self._respawn_player()
                else:
                    terminated = True

        if not self.enemy_formation.enemies:
            reward += 50.0
            
            if self.enemy_respawns_remaining > 0:
                self.enemy_respawns_remaining -= 1
                self._respawn_enemies()
                reward += 20.0
            else:
                reward += 100.0
                terminated = True

        shooting_prob = 0.05 + (0.05 * (1.0 - len(self.enemy_formation.enemies) / (self.difficulty_settings["enemy_rows"] * self.difficulty_settings["enemy_cols"])))
        if self.enemy_formation.enemies and random.random() < shooting_prob:
            enemy = self.enemy_formation.get_lowest_enemy()
            if enemy:
                b = Bullet(enemy.center_x, enemy.center_y, -1)
                self.bullets.append(b)

        if self.player_alive:
            edge_threshold = self.width * 0.1
            if self.player.center_x < edge_threshold or self.player.center_x > self.width - edge_threshold:
                reward -= 0.03

        reward -= 0.01

        obs = self._obs()
        
        return obs, float(reward), bool(terminated), bool(truncated), {
            "enemies_killed": self.enemies_killed,
            "step": self.current_step,
            "player_respawns_remaining": self.player_respawns_remaining,
            "enemy_respawns_remaining": self.enemy_respawns_remaining
        }
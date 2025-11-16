import arcade, time, random, numpy as np

from stable_baselines3 import PPO

from utils.constants import PLAYER_SPEED, BULLET_SPEED, BULLET_RADIUS, PLAYER_ATTACK_SPEED
from utils.preload import player_texture, enemy_texture

class Bullet(arcade.Sprite):
    def __init__(self, x, y, direction_y):
        super().__init__(arcade.texture.make_circle_texture(BULLET_RADIUS, arcade.color.YELLOW), center_x=x, center_y=y)

        self.direction_y = direction_y

    def update(self):
        self.center_y += self.direction_y * BULLET_SPEED

class EnemyFormation():
    def __init__(self, start_x, start_y, spritelist: arcade.SpriteList | None, rows, cols):
        self.grid = [[] for _ in range(rows)]
        self.start_x = start_x
        self.start_y = start_y
        self.rows = rows
        self.cols = cols
        self.spritelist = spritelist

        self.create_formation()

    def create_formation(self, start_x=None, start_y=None):
        if start_x:
            self.start_x = start_x
        if start_y:
            self.start_y = start_y

        del self.grid
        self.grid = [[] for _ in range(self.rows)]

        for row in range(self.rows):
            for col in range(self.cols):
                enemy_sprite = arcade.Sprite(enemy_texture, center_x=self.start_x + col * 100, center_y=self.start_y - row * 100)
                
                if self.spritelist:
                    self.spritelist.append(enemy_sprite)

                self.grid[row].append(enemy_sprite)

    def remove_enemy(self, enemy):
        if self.spritelist and enemy not in self.spritelist:
            return

        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == enemy:
                    self.grid[row][col] = None
                    if self.spritelist:
                        self.spritelist.remove(enemy)
                    return 
    
    def get_lowest_enemy(self):
        valid_cols = []

        for col in range(self.cols):
            row = self.rows - 1
            
            while row >= 0 and self.grid[row][col] is None:
                row -= 1

            if row >= 0:
                valid_cols.append((col, row))

        if not valid_cols:
            return None

        col, row = random.choice(valid_cols)
        return self.grid[row][col]

    def move(self, width, height, direction_type, value):
        if direction_type == "x":
            wall_hit = False
            for enemy in self.enemies:
                self.start_x += value
                enemy.center_x += value

                if enemy.center_x + enemy.width / 2 > width or enemy.center_x < enemy.width / 2:
                    wall_hit = True
            
            if wall_hit:
                for enemy in self.enemies:
                    self.start_x -= value
                    enemy.center_x -= value
        else:
            wall_hit = False
            for enemy in self.enemies:
                self.start_x += value
                enemy.center_y += value

                if enemy.center_y + enemy.height / 2 > height or enemy.center_y < enemy.height / 2:
                    wall_hit = True
            
            if wall_hit:
                for enemy in self.enemies:
                    self.start_y -= value
                    enemy.center_y -= value

    @property
    def center_x(self):
        return self.start_x + (self.cols / 2) * 100

    @property
    def enemies(self):
        return [col for row in self.grid for col in row if not col == None]

class Player(arcade.Sprite): # Not actually the player
    def __init__(self, x, y):
        super().__init__(player_texture, center_x=x, center_y=y)

        self.last_target_change = time.perf_counter()
        self.last_shoot = time.perf_counter()
        self.shoot = False
        self.player_speed = 0

    def update(self, model: PPO, enemy_formation, bullets, width, height, player_respawns_norm, enemy_respawns_norm):
        if enemy_formation.enemies:
            nearest_enemy = min(enemy_formation.enemies, key=lambda e: abs(e.center_x - self.center_x))
            enemy_x = (nearest_enemy.center_x - self.center_x) / width
            enemy_y = (nearest_enemy.center_y - self.center_y) / height
        else:
            enemy_x = 2
            enemy_y = 2

        enemy_count = len(enemy_formation.enemies) / float(max(1, enemy_formation.rows * enemy_formation.cols))
        player_x_norm = self.center_x / width

        curr_bullet = min(bullets, key=lambda b: abs(b.center_x - self.center_x) + abs(b.center_y - self.center_y)) if bullets else None
        if curr_bullet is not None:
            curr_bx = (curr_bullet.center_x - self.center_x) / float(width)
            curr_by = (curr_bullet.center_y - self.center_y) / float(height)
        else:
            curr_bx = 2.0
            curr_by = 2.0

        lowest = max(enemy_formation.enemies, key=lambda e: e.center_y) if enemy_formation.enemies else None
        if lowest is not None:
            lowest_dy = (lowest.center_y - self.center_y) / float(height)
        else:
            lowest_dy = 2.0

        enemy_dispersion = 0.0
        if enemy_formation.enemies:
            xs = np.array([e.center_x for e in enemy_formation.enemies], dtype=np.float32)
            enemy_dispersion = float(xs.std()) / float(width)

        obs = np.array([
            player_x_norm, 
            enemy_x, 
            enemy_y,
            lowest_dy,
            curr_bx, 
            curr_by,
            self.player_speed,
            enemy_count,
            enemy_dispersion,
            time.perf_counter() - self.last_shoot >= PLAYER_ATTACK_SPEED,
            player_respawns_norm, 
            enemy_respawns_norm
        ], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)

        self.prev_x = self.center_x
        if action == 0:
            self.center_x -= PLAYER_SPEED
        elif action == 1:
            self.center_x += PLAYER_SPEED
        elif action == 2:
            pass
        elif action == 3:
            t = time.perf_counter()
            if t - self.last_shoot >= PLAYER_ATTACK_SPEED:
                self.last_shoot = t
                self.shoot = True

        self.player_speed = (self.center_x - self.prev_x) / max(1e-6, PLAYER_SPEED)

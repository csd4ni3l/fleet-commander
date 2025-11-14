import arcade, arcade.gui, random, time

from utils.constants import button_style, ENEMY_ROWS, ENEMY_COLS, PLAYER_ATTACK_SPEED
from utils.preload import button_texture, button_hovered_texture

from game.sprites import Enemy, Player, Bullet

class Game(arcade.gui.UIView):
    def __init__(self, pypresence_client):
        super().__init__()

        self.pypresence_client = pypresence_client
        self.pypresence_client.update(state="Invading Space")

        self.anchor = self.add_widget(arcade.gui.UIAnchorLayout(size_hint=(1, 1)))
        
        self.spritelist = arcade.SpriteList()

        self.player = Player(100, 100)  # not actually player
        self.spritelist.append(self.player)

        self.last_player_shoot = time.perf_counter() # not actually player
        
        self.enemies: list[Enemy] = []
        self.bullets: list[Bullet] = []

        self.summon_enemies()

    def on_show_view(self):
        super().on_show_view()

    def summon_enemies(self):
        enemy_start_x = self.window.width * 0.15
        enemy_start_y = self.window.height * 0.9

        for row in range(ENEMY_ROWS):
            for col in range(ENEMY_COLS):
                enemy_sprite = Enemy(enemy_start_x + col * 100, enemy_start_y - row * 100)
                self.spritelist.append(enemy_sprite)
                self.enemies.append(enemy_sprite)

    def on_update(self, delta_time):
        for enemy in self.enemies:
            enemy.update()

        bullets_to_remove = []

        for bullet in self.bullets:
            bullet.update()
            
            bullet_hit = False
            if bullet.direction_y == 1:
                for enemy in self.enemies:
                    if bullet.rect.intersection(enemy.rect):
                        self.spritelist.remove(enemy)
                        self.enemies.remove(enemy)
                        self.player.target = None
                        bullets_to_remove.append(bullet)
                        bullet_hit = True
                        break
            else:
                if bullet.rect.intersection(self.player.rect):
                    bullets_to_remove.append(bullet)
                    bullet_hit = True
                    
            if not bullet_hit and bullet.center_y > self.window.height or bullet.center_y < 0:
                bullets_to_remove.append(bullet)

        for bullet_to_remove in bullets_to_remove:
            self.spritelist.remove(bullet_to_remove)
            self.bullets.remove(bullet_to_remove)

        self.player.update(self.enemies) # not actually player

        if time.perf_counter() - self.last_player_shoot >= PLAYER_ATTACK_SPEED:
            self.last_player_shoot = time.perf_counter()
            self.shoot(self.player.center_x, self.player.center_y, 1)

    def shoot(self, x, y, direction_y):
        bullet = Bullet(x, y, direction_y)
        self.spritelist.append(bullet)
        self.bullets.append(bullet)

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.SPACE:
            enemy = random.choice(self.enemies)
            self.shoot(enemy.center_x, enemy.center_y, -1)

    def on_draw(self):
        super().on_draw()

        self.spritelist.draw()
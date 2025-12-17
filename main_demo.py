import pygame, time, numpy as np
from env import SimpleArtilleryEnv

SCREEN_W, SCREEN_H = 800, 700
GROUND_Y = 600

def draw_scene(screen, env, shell_pos=None, hit=False):
    screen.fill((28, 30, 34))
    pygame.draw.rect(screen, (50, 50, 50), (0, GROUND_Y, SCREEN_W, SCREEN_H - GROUND_Y))
    base_w, base_h = 60, 40
    target_rect = pygame.Rect(int(env.shelter_x - base_w/2), int(env.shelter_y - base_h/2), base_w, base_h)
    pygame.draw.rect(screen, (200, 80, 80), target_rect)
    pygame.draw.rect(screen, (180, 180, 180), (80, GROUND_Y - 20, 40, 40))
    angle_rad = -env.gun_angle * np.pi / 180.0
    barrel_len = 70
    x0, y0 = 100, GROUND_Y
    x1 = int(x0 + barrel_len * np.cos(angle_rad))
    y1 = int(y0 + -barrel_len * np.sin(angle_rad))
    pygame.draw.line(screen, (200, 180, 90), (x0, y0), (x1, y1), 10)
    if shell_pos is not None:
        sx, sy = int(shell_pos[0]), int(shell_pos[1])
        if 0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H:
            pygame.draw.circle(screen, (220, 70, 70), (sx, sy), 8)
    font = pygame.font.SysFont(None, 24)
    wind_text = font.render(f"Wind: {env.wind:.2f}", True, (200,200,200))
    screen.blit(wind_text, (10, 10))
    if hit:
        font2 = pygame.font.SysFont(None, 72)
        text = font2.render("HIT!", True, (255, 40, 40))
        text_rect = text.get_rect(center=(SCREEN_W // 2, 100))
        screen.blit(text, text_rect)
    pygame.display.flip()

def animate_trajectory(screen, env, trajectory, hit):
    clock = pygame.time.Clock()
    for pos in trajectory:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        draw_scene(screen, env, shell_pos=pos, hit=False)
        clock.tick(33)
    draw_scene(screen, env, shell_pos=trajectory[-1] if len(trajectory) > 0 else None, hit=hit)
    pygame.time.wait(800 if hit else 600)
    return True

def interactive_demo():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Artillery Demo - Arrow keys to change angle, SPACE to fire")
    clock = pygame.time.Clock()
    env = SimpleArtilleryEnv()
    obs, _ = env.reset()
    running = True
    last_fire_time = 0.0
    fire_cooldown_ms = 250
    current_speed = env.default_v
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        angle_change = 0.0
        if keys[pygame.K_LEFT]:
            angle_change = -2.0
        elif keys[pygame.K_RIGHT]:
            angle_change = 2.0
        if keys[pygame.K_UP]:
            current_speed = min(140.0, current_speed + 1.0)
        elif keys[pygame.K_DOWN]:
            current_speed = max(40.0, current_speed - 1.0)
        fired = keys[pygame.K_SPACE]
        if fired and (pygame.time.get_ticks() - last_fire_time) > fire_cooldown_ms:
            last_fire_time = pygame.time.get_ticks()
            obs, reward, terminated, truncated, info = env.step([angle_change, current_speed])
            ok = animate_trajectory(screen, env, info["trajectory"], info["hit"])
            if not ok:
                running = False
            if info["hit"]:
                time.sleep(0.3)
                obs, _ = env.reset()
                current_speed = env.default_v
        else:
            if abs(angle_change) > 0.0:
                env.gun_angle = float(np.clip(env.gun_angle + angle_change, 0.0, 90.0))
            draw_scene(screen, env, shell_pos=None, hit=False)
            clock.tick(30)
    pygame.quit()

def test_random_render(n_episodes=3):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    for ep in range(n_episodes):
        env = SimpleArtilleryEnv()
        obs, _ = env.reset()
        done = False
        shots = 0
        while not done and shots < 20:
            action = [float(np.random.uniform(-3.0, 3.0)), float(np.random.uniform(60.0, 120.0))]
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ok = animate_trajectory(screen, env, info["trajectory"], info["hit"])
            if not ok:
                pygame.quit()
                return
            shots += 1
            pygame.time.wait(120)
        print(f"[Random] Episode {ep+1}: Hit={info['hit']}, Shots={shots}")
        pygame.time.wait(500)
    pygame.quit()

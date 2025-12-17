import argparse
import numpy as np
import time
import pygame
from stable_baselines3 import SAC

from env import SimpleArtilleryEnv
from dqn_wrapper import DiscreteActionWrapper
from wind_obs_wrapper import WindObsWrapper
from main_demo import animate_trajectory


# ================================================================
# TRAIN SAC
# ================================================================
def train_sac(total_timesteps=600_000, model_path="sac_artillery.zip"):
    print("\n🚀 Starting SAC Training...\n")

    # Base env
    env = SimpleArtilleryEnv()

    # Wind normalization
    env = WindObsWrapper(env)

    # Discretized actions
    angle_changes = np.linspace(-5.0, 5.0, 9)
    speed_values = np.linspace(60.0, 120.0, 7)
    env_wrapped = (env, angle_changes, speed_values)

    # SAC Agent
    model = SAC(
        policy="MlpPolicy",
        env=env_wrapped,
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)

    print(f"\n✅ SAC Model saved to {model_path}\n")


# ================================================================
# TEST SAC
# ================================================================
def test_sac(model_path="sac_artillery.zip", n_episodes=5):
    print("\n🎯 Running SAC Test Episodes...\n")

    pygame.init()
    screen = pygame.display.set_mode((800, 700))
    pygame.display.set_caption("SAC Artillery Test")

    # Same wrappers as training
    angle_changes = np.linspace(-5.0, 5.0, 9)
    speed_values = np.linspace(60.0, 120.0, 7)

    env = DiscreteActionWrapper(
        WindObsWrapper(SimpleArtilleryEnv()),
        angle_changes,
        speed_values
    )

    # Load model
    model = SAC.load(model_path)

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        shots = 0

        while not done and shots < 15:

            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # Animate shot
            traj = info.get("trajectory", [])
            hit = info.get("hit", False)
            animate_trajectory(screen, env.unwrapped, traj, hit)

            obs = next_obs
            shots += 1
            time.sleep(0.1)

        print(f"[SAC TEST] Episode {ep+1}: Hit={info.get('hit')}, Shots={shots}")
        time.sleep(0.4)

    pygame.quit()
    print("\n🏁 SAC Testing Complete\n")


# ================================================================
# MAIN — ARGPARSE
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test SAC Artillery Agent")

    parser.add_argument("--train", action="store_true", help="Train the SAC model")
    parser.add_argument("--test", action="store_true", help="Test the SAC model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")

    args = parser.parse_args()

    MODEL_PATH = "sac_artillery.zip"

    # User selected: TRAIN
    if args.train:
        train_sac()

    # User selected: TEST
    if args.test:
        test_sac(model_path=MODEL_PATH, n_episodes=args.episodes)

    # If nothing selected → show help
    if not args.train and not args.test:
        parser.print_help()

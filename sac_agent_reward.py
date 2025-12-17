# sac_agent_clean.py
import argparse
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import pygame

from env import SimpleArtilleryEnv   # your env.py file
from main_demo import animate_trajectory  # must exist and accept (screen, env, traj, hit)

def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


# -------------------------
# Reward shaping wrapper
# -------------------------
class RewardShapingWrapper(gym.Wrapper):
    """
    Apply additional shaping on top of the base env reward.
    Keeps same interface as gym.Env so it composes with Monitor/DummyVecEnv.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Basic shaping: high positive for hit, penalty proportional to distance otherwise,
        # small per-shot penalty to encourage 1-shot behavior.
        shaped = base_reward

        # If env sets shell_pos and shelter coords, use them
        shell_pos = getattr(self.env, "shell_pos", None)
        shelter_x = getattr(self.env, "shelter_x", None)
        shelter_y = getattr(self.env, "shelter_y", None)

        if shell_pos is not None and shelter_x is not None:
            dx = float(shell_pos[0] - shelter_x)
            dy = float(shell_pos[1] - shelter_y)
            dist = np.hypot(dx, dy)
            # give more reward for closer shots (non-linear)
            shaped += 200.0 * (1.0 / (1.0 + dist))
        else:
            # fallback: small shaping using base reward sign
            shaped += 0.0

        # Extra hit bonus (if env didn't already give a big reward)
        if info.get("hit", False):
            shaped += 300.0

        # small penalty per shot to encourage single-shot solves
        shaped -= 3.0

        # optional: penalize strong winds slightly to encourage compensation
        wind = info.get("wind", 0.0)
        shaped -= 0.1 * abs(wind)

        return obs, float(shaped), terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# -------------------------
# Observation normalization wrapper
# -------------------------
class NormalizeObsWrapper(gym.ObservationWrapper):
    """
    Normalize observations to [0,1] using env.observation_space bounds.
    Works with single-env (non-vectorized) and also underlying Monitor/DummyVecEnv.
    """
    def __init__(self, env):
        super().__init__(env)
        self.low = env.observation_space.low.astype(np.float32)
        self.high = env.observation_space.high.astype(np.float32)

    def observation(self, obs):
        # ensure float32
        obs = np.asarray(obs, dtype=np.float32)
        return (obs - self.low) / (self.high - self.low + 1e-8)


# -------------------------
# Make environment factory for vectorized training
# -------------------------
def make_env(seed=None):
    def _init():
        env = SimpleArtilleryEnv()
        env = RewardShapingWrapper(env)
        env = NormalizeObsWrapper(env)
        env = Monitor(env)  # records episode stats
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


# -------------------------
# Train SAC
# -------------------------
def train_sac(model_path="sac_artillery.zip", total_timesteps=300_000, seed=0):
    # Create a DummyVecEnv with one env (SB3 requires a VecEnv for off-policy algorithms)
    env = DummyVecEnv([make_env(seed)])

    # Sanity check action/obs spaces
    single_env = env.envs[0].env  # Monitor -> underlying env
    print("Action space:", single_env.action_space)
    print("Observation space:", single_env.observation_space)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1,
        seed=seed
    )

    print(f"\n🚀 Training SAC for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"\n✅ Saved SAC model to '{model_path}'")
    env.close()



# -------------------------
# Test SAC (non-vectorized, with rendering)
# -------------------------
def test_sac(model_path="sac_artillery.zip", n_episodes=5, render=True):
    pygame.init()
    screen = pygame.display.set_mode((800, 700))
    pygame.display.set_caption("SAC Artillery Test")

    # Create the same wrappers as used in training (except Monitor / VecEnv)
    env = SimpleArtilleryEnv()
    env = RewardShapingWrapper(env)
    env = NormalizeObsWrapper(env)

    model = SAC.load(model_path)

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < 20:
            # model.predict expects the same observation format as used in training
            action, _ = model.predict(obs, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # animate trajectory collected in info
            traj = info.get("trajectory", [])
            hit = info.get("hit", False)

            # If your animate_trajectory expects (screen, env, traj, hit) adjust accordingly:
            try:
                # animate_trajectory(screen, env, traj, hit)
                base_env = unwrap_env(env)
                animate_trajectory(screen, base_env, traj, hit)

            except TypeError:
                # fallback if animate_trajectory signature is different
                animate_trajectory(screen, traj, hit)

            obs = next_obs
            steps += 1
            time.sleep(0.08)

            # handle pygame events so window remains responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

        print(f"[SAC TEST] Episode {ep+1}: Hit={info.get('hit')}, Steps={steps}")
        time.sleep(0.4)

    pygame.quit()


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC agent (clean) for artillery")
    parser.add_argument("--train", action="store_true", help="Train SAC model")
    parser.add_argument("--test", action="store_true", help="Test SAC model")
    parser.add_argument("--timesteps", type=int, default=300000, help="Timesteps for training")
    parser.add_argument("--model", type=str, default="sac_artillery.zip", help="Model path")
    parser.add_argument("--episodes", type=int, default=5, help="Test episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    if args.train:
        train_sac(model_path=args.model, total_timesteps=args.timesteps, seed=args.seed)

    if args.test:
        test_sac(model_path=args.model, n_episodes=args.episodes)

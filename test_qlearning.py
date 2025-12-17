# test_qlearning.py
import pickle
import numpy as np
import pygame
import time
from env import SimpleArtilleryEnv
from main_demo import animate_trajectory  # make sure this works with your env

class TestQLearning:
    def __init__(self):
        # Initialize environment
        self.env = SimpleArtilleryEnv()

        # Load trained Q-table and actions
        with open("simple_qtable.pkl", "rb") as f:
            self.q_table = pickle.load(f)
        with open("simple_actions.pkl", "rb") as f:
            self.action_list = pickle.load(f)

        # Ensure action list length matches Q-table
        assert len(self.action_list) == self.q_table.shape[3], "Action list length must match Q-table"

        # Parameters must match training
        self.n_angle_bins = self.q_table.shape[0]
        self.n_dist_bins = self.q_table.shape[1]
        self.n_wind_bins = self.q_table.shape[2]
        self.angle_space = np.linspace(0.0, 90.0, self.n_angle_bins)
        self.dist_space = np.linspace(0.0, 800.0, self.n_dist_bins)
        # Must match training exactly
        self.wind_space = np.arange(-20, 21)

        # Initialize Pygame for visualization
        pygame.init()
        self.screen = pygame.display.set_mode((800, 700))
        pygame.display.set_caption("Q-Learning Test Agent")

    def discretize(self, obs):
        gun_angle = float(obs[0])
        shelter_x = float(obs[1])
        wind = float(obs[3]) if len(obs) > 3 else 0.0
        dist = max(0.0, shelter_x - 100.0)

        angle_idx = int(np.digitize(gun_angle, self.angle_space) - 1)
        angle_idx = np.clip(angle_idx, 0, self.n_angle_bins - 1)

        dist_idx = int(np.digitize(dist, self.dist_space) - 1)
        dist_idx = np.clip(dist_idx, 0, self.n_dist_bins - 1)

        wind_idx = int(np.digitize(wind, self.wind_space) - 1)
        wind_idx = np.clip(wind_idx, 0, self.n_wind_bins - 1)

        return angle_idx, dist_idx, wind_idx

    def run(self, n_episodes=10):
        one_shot_hits = 0
        total_hits = 0
        total_steps = 0
        
        print("\n🎯 Running Q-Learning Test Episodes...\n")
        
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            steps = 0
            total_reward = 0.0

            while not done and steps < 20:
                # Handle Pygame events (so window stays responsive)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        print("\n✅ Q-Learning Testing Complete")
                        return

                # Discretize state including wind
                angle_idx, dist_idx, wind_idx = self.discretize(obs)

                # Choose greedy action
                action_idx = int(np.argmax(self.q_table[angle_idx, dist_idx, wind_idx]))
                action_idx = np.clip(action_idx, 0, len(self.action_list) - 1)
                action = self.action_list[action_idx]

                # Step in environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1

                # Animate shell trajectory
                traj = info.get("trajectory", [])
                hit = info.get("hit", False)
                animate_trajectory(self.screen, self.env, traj, hit)
                time.sleep(0.1)  # small delay to see movement

            # Track statistics
            if info['hit']:
                total_hits += 1
                if steps == 1:
                    one_shot_hits += 1
            total_steps += steps
            
            print(f"Episode {ep+1}: steps={steps}, hit={info['hit']}, reward={total_reward:.1f}")
            time.sleep(0.5)  # pause between episodes

        # Print summary
        avg_steps = total_steps / n_episodes
        hit_rate = (total_hits / n_episodes) * 100
        one_shot_rate = (one_shot_hits / n_episodes) * 100
        
        print("\n" + "="*50)
        print("📊 Q-Learning Test Results Summary")
        print("="*50)
        print(f"Total hits:     {total_hits}/{n_episodes} ({hit_rate:.1f}%)")
        print(f"One-shot hits:  {one_shot_hits}/{n_episodes} ({one_shot_rate:.1f}%)")
        print(f"Average steps:  {avg_steps:.2f}")
        print("="*50 + "\n")
        
        pygame.quit()
        print("✅ Q-Learning Testing Complete")


if __name__ == "__main__":
    tester = TestQLearning()
    tester.run(n_episodes=5)
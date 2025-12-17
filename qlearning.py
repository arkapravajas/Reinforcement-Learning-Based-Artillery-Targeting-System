import numpy as np
import pickle
from env import SimpleArtilleryEnv

class SimpleQLearning:
    def __init__(self,
                 n_angle_bins=60,  # Increased from 36
                 n_dist_bins=40,   # Increased from 24
                 n_angle_change_bins=21,  # Increased from 15
                 n_speed_bins=21,  # Increased from 15
                 alpha=0.25,  # Lower learning rate for stability
                 gamma=0.80,  # Higher discount factor
                 epsilon=1.0, 
                 epsilon_decay=0.9975,  # Slower decay
                 epsilon_min=0.00):
        self.env = SimpleArtilleryEnv()

        # Finer discretization
        self.n_angle_bins = n_angle_bins
        self.n_dist_bins = n_dist_bins
        self.angle_space = np.linspace(0.0, 90.0, n_angle_bins)
        self.dist_space = np.linspace(0.0, 800.0, n_dist_bins)
        
        # Wind: keep integer bins
        self.wind_space = np.arange(-20, 21)
        self.n_wind_bins = len(self.wind_space)

        # Finer action discretization
        self.angle_changes = np.linspace(-10.0, 10.0, n_angle_change_bins)
        self.speed_values = np.linspace(40.0, 140.0, n_speed_bins)
        self.action_list = [(float(ac), float(sv)) 
                           for ac in self.angle_changes 
                           for sv in self.speed_values]
        self.n_action_bins = len(self.action_list)

        # Initialize Q-table
        self.q_table = np.ones((n_angle_bins, n_dist_bins, 
                                self.n_wind_bins, self.n_action_bins), dtype=float)* 300.0

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Track statistics
        self.hit_count = 0
        self.episode_count = 0

    def discretize(self, obs):
        """Discretize observation to Q-table indices."""
        gun_angle = float(obs[0])
        shelter_x = float(obs[1])
        wind = float(obs[3])
        dist = max(0.0, shelter_x - 100.0)

        angle_idx = int(np.digitize(gun_angle, self.angle_space) - 1)
        angle_idx = np.clip(angle_idx, 0, self.n_angle_bins - 1)

        dist_idx = int(np.digitize(dist, self.dist_space) - 1)
        dist_idx = np.clip(dist_idx, 0, self.n_dist_bins - 1)

        wind_idx = int(np.digitize(wind, self.wind_space) - 1)
        wind_idx = np.clip(wind_idx, 0, self.n_wind_bins - 1)

        return angle_idx, dist_idx, wind_idx

    def compute_shaped_reward(self, obs, base_reward, info, steps):
        """Ultra-optimized for 95%+ one-shot accuracy."""
        shaped = base_reward
        
        # Get positions
        shell_pos = info.get('trajectory', [])[-1] if info.get('trajectory') else (100.0, 600.0)
        shelter_x = float(obs[1])
        shelter_y = float(obs[2])
        dx = float(shell_pos[0] - shelter_x)
        dy = float(shell_pos[1] - shelter_y)
        dist = np.hypot(dx, dy)
        
        # TUNED VALUES FOR MAXIMUM ONE-SHOT ACCURACY
        shaped += 400.0 * (1.0 / (1.0 + dist))  # Strong proximity reward
        
        if info.get("hit", False):
            shaped += 450.0  # Base hit (slightly reduced)
            if steps == 1:
                shaped += 600.0  # MASSIVE first-shot bonus
        
        shaped -= 8.0 * (steps ** 1.5)  # Heavy multi-shot penalty
        if steps > 0:
            shaped -= 200.0

        wind = float(obs[3])
        shaped -= 0.2 * abs(wind)  # Increased wind penalty
        
        gun_angle = float(obs[0])
        ideal_angle = 45.0 + (wind * 0.35)  # Better wind compensation
        angle_error = abs(gun_angle - ideal_angle)
        shaped -= 0.1 * angle_error  # Stronger angle guidance
        
        return float(shaped)

    def select_action(self, angle_idx, dist_idx, wind_idx):
        """Epsilon-greedy with decaying exploration."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_action_bins)
        return int(np.argmax(self.q_table[angle_idx, dist_idx, wind_idx]))

    def train(self, n_episodes=500000, max_steps_per_episode=3, verbose=True):
        """Train with reward shaping and better exploration."""
        print(f"\n🚀 Starting Q-Learning Training with reward shaping...")
        print(f"   Episodes: {n_episodes}")
        print(f"   State space: {self.n_angle_bins}×{self.n_dist_bins}×{self.n_wind_bins} = {self.n_angle_bins * self.n_dist_bins * self.n_wind_bins:,}")
        print(f"   Action space: {self.n_action_bins} ({len(self.angle_changes)}×{len(self.speed_values)})")
        print(f"   Total Q-table size: {self.q_table.size:,} entries\n")
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            steps = 0
            episode_reward = 0.0

            while not done and steps < max_steps_per_episode:
                # Discretize state
                angle_idx, dist_idx, wind_idx = self.discretize(obs)

                # Select action
                action_idx = self.select_action(angle_idx, dist_idx, wind_idx)
                action_value = self.action_list[action_idx]

                # Take step
                next_obs, base_reward, terminated, truncated, info = self.env.step(action_value)
                done = terminated or truncated
                
                # Apply reward shaping
                reward = self.compute_shaped_reward(next_obs, base_reward, info, steps)
                episode_reward += reward

                # Discretize next state
                next_angle_idx, next_dist_idx, next_wind_idx = self.discretize(next_obs)

                # Q-learning update
                best_next = np.max(self.q_table[next_angle_idx, next_dist_idx, next_wind_idx])
                td_target = reward + (0 if done else self.gamma * best_next)
                td_error = td_target - self.q_table[angle_idx, dist_idx, wind_idx, action_idx]
                self.q_table[angle_idx, dist_idx, wind_idx, action_idx] += self.alpha * td_error

                obs = next_obs
                steps += 1

            # Track hits
            if info.get('hit', False):
                self.hit_count += 1
            self.episode_count += 1

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Logging
            if verbose and (episode % 1000 == 0):
                hit_rate = self.hit_count / max(1, self.episode_count) * 100
                print(f"[Q] Episode {episode:,}, epsilon={self.epsilon:.4f}, "
                      f"hit_rate={hit_rate:.1f}%, last_reward={episode_reward:.1f}")
                # Reset counters every 10k episodes for recent performance
                if episode % 10000 == 0 and episode > 0:
                    self.hit_count = 0
                    self.episode_count = 0


        # 🔒 Lock greedy policy
        self.epsilon = 0.0

        # Save
        with open("simple_qtable.pkl", "wb") as f:
            pickle.dump(self.q_table, f)
        with open("simple_actions.pkl", "wb") as f:
            pickle.dump(self.action_list, f)

        if verbose:
            print("\n✅ Q-Learning training complete!")
            print(f"   Q-table saved to simple_qtable.pkl")
            print(f"   Actions saved to simple_actions.pkl\n")


if __name__ == "__main__":
    agent = SimpleQLearning()
    agent.train(n_episodes=500000, verbose=True)
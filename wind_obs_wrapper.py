import gymnasium as gym
import numpy as np

class WindObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wind_space = np.arange(-20, 21)  # [-20..20], 41 bins

        # Original observation space size
        orig_low = self.observation_space.low
        orig_high = self.observation_space.high

        # Append one extra dimension: wind_idx ∈ [0, 40]
        new_low = np.append(orig_low, 0)
        new_high = np.append(orig_high, 40)

        self.observation_space = gym.spaces.Box(
            low=new_low,
            high=new_high,
            dtype=float
        )

    def observation(self, obs):
        wind = float(obs[3])  # real wind from env
        wind_idx = int(np.digitize(wind, self.wind_space) - 1)
        wind_idx = np.clip(wind_idx, 0, len(self.wind_space) - 1)

        # Append to observation
        return np.append(obs, wind_idx)

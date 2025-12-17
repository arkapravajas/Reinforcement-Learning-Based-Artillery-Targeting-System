import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleArtilleryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Actions: [angle_change_deg, muzzle_velocity]
        self.action_space = spaces.Box(low=np.array([-10.0, 40.0], dtype=np.float32),
                                       high=np.array([10.0, 140.0], dtype=np.float32),
                                       dtype=np.float32)
        # Observation: [gun_angle (0..90), shelter_x (0..800), shelter_y (0..700), wind (-40..40)]
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, -40.0], dtype=np.float32),
                                            high=np.array([90.0, 800.0, 700.0, 40.0], dtype=np.float32),
                                            dtype=np.float32)
        self.default_v = 80.0
        self.g = 9.81
        self.ground_y = 600.0
        self.current_step = 0
        self.max_steps = 20

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.gun_angle = 45.0
        self.shelter_x = float(self.np_random.integers(400, 700))
        self.shelter_y = 600.0
        self.wind = float(self.np_random.uniform(-20.0, 20.0))
        self.trajectory = []
        self.last_hit = False
        self.current_step = 0
        self.shell_pos = None
        obs = np.array([self.gun_angle, self.shelter_x, self.shelter_y, self.wind], dtype=np.float32)
    

        info = {}
        return obs, info

    def step(self, action):
        # action: array-like [angle_change, muzzle_v]
        arr = np.array(action, dtype=float)
        angle_change = float(arr[0])
        muzzle_v = float(arr[1]) if len(arr) > 1 else self.default_v
        # clamp angle
        self.gun_angle = float(np.clip(self.gun_angle + angle_change, 0.0, 90.0))
        angle_rad = np.deg2rad(self.gun_angle)
        vx = muzzle_v * np.cos(angle_rad)
        vy = muzzle_v * np.sin(angle_rad)
        wind = self.wind
        vx_total = vx + wind
        t_flight = max(0.05, 2.0 * vy / self.g)
        dt = 0.03
        times = np.arange(0.0, t_flight + dt, dt)

        traj = []
        hit = False
        hit_threshold = 30.0
        for t in times:
            x = 100.0 + vx_total * t
            y_phys = vy * t - 0.5 * self.g * t * t
            y = self.ground_y - y_phys
            traj.append((float(x), float(y)))
            if np.hypot(x - self.shelter_x, y - self.shelter_y) <= hit_threshold:
                hit = True
                break

        self.trajectory = traj
        self.shell_pos = traj[-1] if len(traj) > 0 else (100.0, self.ground_y)
        self.last_hit = bool(hit)
        self.current_step += 1
        terminated = self.last_hit
        truncated = self.current_step >= self.max_steps

        if self.last_hit:
            reward = 500.0
        else:
            dist = float(np.hypot(self.shell_pos[0] - self.shelter_x, self.shell_pos[1] - self.shelter_y))
            reward = -dist 

        obs = np.array([self.gun_angle, self.shelter_x, self.shelter_y, self.wind], dtype=np.float32)
        

        info = {"trajectory": self.trajectory, "hit": self.last_hit, "wind": self.wind}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

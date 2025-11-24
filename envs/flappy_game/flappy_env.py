# flappy_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class FlappyEnv(gym.Env):
    """
    Minimal Flappy Bird-style environment.
    State includes:
        - bird y-position
        - bird velocity
        - pipe gap y-position
        - horizontal distance to pipe
    Action space:
        0 = do nothing
        1 = flap upward
    """

    metadata = {"render_modes": []}

    def __init__(self, persona="explorer"):
        super().__init__()
        self.persona = persona

        # Bird physics
        self.gravity = 0.5
        self.flap_strength = -6.5
        self.max_velocity = 10.0

        # Pipe settings
        self.pipe_gap = 40
        self.pipe_width = 20
        self.pipe_distance_reset = 70
        self.pipe_speed = 2

        self.observation_space = spaces.Box(
            low=np.array([0.0, -12.0, 0.0, 0.0]),
            high=np.array([200.0, 12.0, 200.0, 200.0]),
            dtype=np.float32,
        )

        # Actions: 0 = no flap, 1 = flap
        self.action_space = spaces.Discrete(2)

        self.max_steps = 500
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bird_y = 100.0
        self.velocity = 0.0

        self.pipe_gap_y = random.randint(40, 160)
        self.pipe_x = 200

        self.steps = 0
        self.pipes_passed = 0
        self.last_action = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [
                self.bird_y,
                self.velocity,
                self.pipe_gap_y,
                self.pipe_x
            ],
            dtype=np.float32
        )

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Apply action
        if action == 1:
            self.velocity = self.flap_strength
        else:
            self.velocity += self.gravity

        # Clamp velocity
        self.velocity = np.clip(self.velocity, -12, 12)

        # Update position
        self.bird_y += self.velocity

        # Move pipe
        self.pipe_x -= self.pipe_speed

        # Pipe reset when passed
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.pipe_distance_reset
            self.pipe_gap_y = random.randint(40, 160)
            self.pipes_passed += 1

        hit_ground = self.bird_y <= 0
        hit_ceiling = self.bird_y >= 200

        hit_pipe = (
            self.pipe_x < 20  # bird x-position is approximated as 20px
            and self.pipe_x + self.pipe_width > 0
            and not (self.pipe_gap_y - self.pipe_gap <= self.bird_y <= self.pipe_gap_y + self.pipe_gap)
        )

        if hit_ground or hit_ceiling or hit_pipe:
            terminated = True
            info["death_reason"] = (
                "ground" if hit_ground else
                "ceiling" if hit_ceiling else
                "pipe"
            )

        if self.persona == "explorer":
            reward += 0.2  # encourages movement forward
            reward += self.pipes_passed * 1.0
            if terminated:
                reward -= 5.0

        elif self.persona == "survivor":
            reward += 0.5  # strong survival reward
            reward += self.pipes_passed * 2.0
            if terminated:
                reward -= 10.0

        # Limit steps
        if self.steps >= self.max_steps:
            truncated = True

        self.last_action = action

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass 

    def close(self):
        pass
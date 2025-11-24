# web_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class WebFlowEnv(gym.Env):
    """
    A simplified web workflow environment:
    login page → profile page → done page.
    Randomized UI faults are introduced to simulate software bugs.
    """

    metadata = {"render_modes": []}

    def __init__(self, persona="explorer"):
        super().__init__()

        self.persona = persona

        # Action meanings:
        # 0 = type username
        # 1 = type password
        # 2 = click login
        # 3 = fill profile
        # 4 = save profile
        # 5 = random click
        # 6 = reset form
        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3 + 1 + 1 + 1,), dtype=np.float32
        )

        self.max_steps = 50
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_page = "login"
        self.has_error = False
        self.num_elements_norm = 0.4
        self.last_action = 0
        self.step_count = 0

        self.visited_pages = {"login"}

        return self._get_obs(), {}

    def _get_obs(self):
        page_vec = {
            "login":  [1.0, 0.0, 0.0],
            "profile":[0.0, 1.0, 0.0],
            "done":   [0.0, 0.0, 1.0]
        }[self.current_page]

        obs = np.array([
            *page_vec,
            1.0 if self.has_error else 0.0,
            self.num_elements_norm,
            float(self.last_action) / (self.action_space.n - 1)
        ], dtype=np.float32)

        return obs

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self.current_page == "login":
            if action == 2:  # click login
                # random login failure
                if random.random() < 0.12:
                    self.has_error = True
                else:
                    self.current_page = "profile"
                    self.has_error = False

        elif self.current_page == "profile":
            if action == 4:  # save profile
                # random backend failure
                if random.random() < 0.10:
                    self.has_error = True
                else:
                    self.current_page = "done"
                    self.has_error = False

        # random clicks
        if action == 5 and random.random() < 0.20:
            self.has_error = True

        # reset
        if action == 6:
            self.current_page = "login"
            self.has_error = False

        page = self.current_page

        if self.persona == "explorer":
            if page not in self.visited_pages:
                reward += 3.0
                self.visited_pages.add(page)
            if self.has_error:
                reward -= 2.0
            if page == "done":
                reward += 10.0

        elif self.persona == "survivor":
            if not self.has_error and page != "done":
                reward += 1.0
            if self.has_error:
                reward -= 8.0
            if page == "done":
                reward += 5.0

        if self.current_page == "done":
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        self.last_action = action

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
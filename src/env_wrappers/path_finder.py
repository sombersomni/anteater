from gymnasium import Wrapper, Env
from typing import Tuple
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class PathFinderRewardWrapper(Wrapper):
    def __init__(
        self,
        env: Env,
        grid_size: int=4,
        goal_position: Tuple[int, int]=(3, 3)
    ):
        super().__init__(env)
        self._visited = set()
        self.grid_size = grid_size
        self.goal_position = (
            (grid_size - 1, grid_size - 1)
            if goal_position is None
            else goal_position
        )
        self.max_steps = grid_size ** 2
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        row, col = obs % self.grid_size, obs // self.grid_size
        agent_current_position = np.array((row, col))
        agent_goal_position = np.array(self.goal_position)
        vistor_reward = (0 if obs in self._visited else 1 / self.env.observation_space.n)
        distance = np.linalg.norm(agent_goal_position - agent_current_position)
        distance_reward = 1 / (distance + 1)
        new_reward = reward + vistor_reward + distance_reward
        self._visited.add(obs)
        info["win_state"] = (
            obs == self.goal_position[0] * self.grid_size + self.goal_position[1]
        )
        return obs, new_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._visited.clear()
        return super().reset(**kwargs)

    def render(self):
        return super().render()
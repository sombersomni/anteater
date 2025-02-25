from gymnasium import Wrapper, Env


class PathFinderRewardWrapper(Wrapper):
    def __init__(
        self,
        env: Env
    ):
        super().__init__(env)
        self._visited = set()
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        vistor_reward = (1 if obs in self._visited else 0)
        exploration_reward = 1 if not terminated else 0
        self._visited.add(obs)
        new_reward = reward + vistor_reward + exploration_reward
        return obs, new_reward, terminated, truncated, info

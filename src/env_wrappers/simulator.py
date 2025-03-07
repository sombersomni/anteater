from gymnasium import Wrapper

from src.utils.logs import setup_logger


logger = setup_logger("Simulator Environment", f"{__name__}.log")


class SimulatorWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def render(self, mode="human"):
        observation = None
        try:
            observation = super().render(mode=mode)
        except Exception as e:
            observation = super().render()
        return observation

    def step(self, action):
        step_tuple = super().step(action)
        if step_tuple is None:
            logger.info("Environment doesn't support step.")
            raise NotImplementedError("Environment doesn't support step.")
        if len(step_tuple) == 5:
            logger.info("We can assume the environment is using terminated and truncated for done state.")
            observation, reward, terminated, truncated, info = step_tuple
            done = terminated or truncated
            return observation, reward, done, info
        logger.info("We can assume the environment is combining terminated and truncated.")
        return step_tuple

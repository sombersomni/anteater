from gym import Env

from src.utils.logging import setup_logger


logger = setup_logger("Simulator Environment", f"{__name__}.log")


class SimulatorEnv(Env):

    def render(self, mode="human"):
        observation = None
        try:
            observation = self.render(mode=mode)
        except Exception as e:
            logger.info("Environment doesn't support rendering on mode. Run rendering normally.")
            observation = self.render()
            if observation is None:
                logger.info("Environment doesn't support rendering.")
        return observation


    def step(self, action):
        step_tuple = self.step(action)
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
 
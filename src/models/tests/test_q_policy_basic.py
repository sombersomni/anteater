import pytest
import gymnasium as gym
from gymnasium import spaces
from src.models.q_policy_basic import QPolicyBasic
from src.models.mocks import FakeGameEnv, QMockPolicy

@pytest.fixture
def env():
    return FakeGameEnv()

def test_protocol(env):
    policy = QPolicyBasic(
        env=env,
        gamma=0.9
    )
    assert isinstance(policy, QPolicyBasic)

def test_get_action_max(env):
    observation = 0
    policy = QPolicyBasic(
        env=env,
        gamma=0.9
    )
    for i in range(4):
        policy.update_reward_state(observation, i, i, False, {})
    action = policy.get_predicted_action(observation, 0)
    assert action == 3

def test_get_reward_default(env):
    policy = QPolicyBasic(
        env=env,
        gamma=0.9
    )
    observation = 0
    action = 1
    reward = policy.get_predicted_reward(observation, action, 1, False, {})
    assert reward == 0  # Check the reward is within the action space

def test_get_max_reward(env):
    observation = 0
    policy = QPolicyBasic(
        env=env,
        gamma=0.9
    )
    for i in range(4):
        policy.update_reward_state(observation, i, 1 if i == 0 else 0, False, {})
    max_reward = policy.get_max_reward(observation)
    assert max_reward == 1

# def test_get_predicted_reward(env):
#     observation = 0
#     next_observation = 1
#     action = 1
#     reward = 0.1  # Example reward
#     done = False
#     info = {}
#     policy = QPolicyBasic(
#         env=env,
#         gamma=0.9
#     )
#     for i in range(4):
#         policy.update_reward_state(observation, i, 1 if i == 0 else 0, False, {})
#     new_reward = policy.get_predicted_reward(
#         next_observation,
#         observation,
#         action,
#         reward,
#         done,
#         info,
#     )
#     assert new_reward == 1
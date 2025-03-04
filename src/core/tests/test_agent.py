import pytest
from unittest.mock import patch, Mock
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete
from src.core.agent import Agent, MemoryPacket, ObservationInfo
from src.models.mocks import QMockPolicy


class MockEnv(Env):
    """
    A mock environment class for testing the Agent.
    """
    def __init__(self, n_actions=4):
        self.action_space = Discrete(n_actions)
        self.observation_space = Discrete(16)


MockEnv.step = Mock(return_value=(0, 1, False, False, {}))
MockEnv.reset = Mock(return_value=None)
MockEnv.render = Mock(return_value=None)


@pytest.fixture
def env():
    return MockEnv(n_actions=4)

@pytest.fixture
def q_policy():
    return QMockPolicy(env=env)


def test_init(env, q_policy):
    """
    Test the initialization of the Agent class.
    """

    agent = Agent(env, policy=q_policy, debug=True)
    assert agent._env == env

def test_set_env(env, q_policy):
    """
    Test the set_env method to ensure the environment can be updated.
    """
    env_other = MockEnv(n_actions=5)
    agent = Agent(env, policy=q_policy, debug=True)
    assert agent._env == env
    agent.set_env(env_other)
    assert agent._env == env_other

def test_select_action_greedy(env, q_policy):
    """
    Test select_action to ensure it selects the greedy action when epsilon=1.
    """
    agent = Agent(env, policy=q_policy, debug=True)
    agent.update(1, 0, 1, 2)
    action = agent.select_action(0, epsilon=1)
    assert action == 2
    assert agent.action == 2


def test_select_action_equal_rewards(env, q_policy):
    """
    Test select_action to ensure it selects the first action when all rewards are equal.
    """
    agent = Agent(env, policy=q_policy, debug=True)
    action = agent.select_action(0, epsilon=1)
    assert action == 0

# def test_update():
#     """
#     Test the update method without future rewards.
#     """
#     env = MockEnv(n_actions=4)
#     agent = Agent(env)
#     agent._rewards_by_action_state[(0, 0)] = 1
#     agent.update(reward=1, observation=0, next_observation=1, action=0, gamma=0.9, epsilon=0.1)
#     assert len(agent.queue) == 1
#     packet = agent.queue[0]
#     assert packet.state == 0
#     assert packet.action == 0
#     assert np.isclose(packet.reward, 0.9)

# def test_update_with_future_reward():
#     """
#     Test the update method considering future rewards.
#     """
#     env = MockEnv(n_actions=4)
#     agent = Agent(env)
#     agent._rewards_by_action_state[(0, 0)] = 1
#     agent._rewards_by_action_state[(1, 1)] = 2
#     agent.update(reward=1, observation=0, next_observation=1, action=0, gamma=0.9, epsilon=0.1)
#     assert len(agent.queue) == 1
#     packet = agent.queue[0]
#     assert packet.state == 0
#     assert packet.action == 0
#     assert np.isclose(packet.reward, 1.08)

# def test_compute_rewards_win():
#     """
#     Test compute_rewards for a winning state.
#     """
#     env = MockEnv(n_actions=4)
#     agent = Agent(env)
#     packet1 = StateActionRewardPacket(state=0, action=0, reward=1.0)
#     packet2 = StateActionRewardPacket(state=1, action=1, reward=2.0)
#     agent.queue = [packet1, packet2]
#     total_rewards = agent.compute_rewards(win_state=True)
#     assert total_rewards == 2.0
#     assert agent._rewards_by_action_state[(1, 1)] == 2.0
#     assert (0, 0) not in agent._rewards_by_action_state or agent._rewards_by_action_state[(0, 0)] == 0
#     assert len(agent.queue) == 0

# def test_compute_rewards_lose():
#     """
#     Test compute_rewards for a losing state.
#     """
#     env = MockEnv(n_actions=4)
#     agent = Agent(env)
#     packet1 = StateActionRewardPacket(state=0, action=0, reward=1.0)
#     packet2 = StateActionRewardPacket(state=1, action=1, reward=2.0)
#     agent.queue = [packet1, packet2]
#     total_rewards = agent.compute_rewards(win_state=False)
#     assert total_rewards == -2.0
#     assert agent._rewards_by_action_state[(1, 1)] == -2.0
#     assert len(agent.queue) == 0

# def test_clear_queue():
#     """
#     Test the clear_queue method to ensure the queue is cleared.
#     """
#     env = MockEnv(n_actions=4)
#     agent = Agent(env)
#     agent.queue = [1, 2, 3]
#     agent.clear_queue()
#     assert agent.queue == []

# def test_str():
#     """
#     Test the string representation of the Agent.
#     """
#     env = MockEnv(n_actions=4)
#     agent = Agent(env)
#     agent.action = 1
#     assert str(agent) == "Agent.v1 | current action: 1"
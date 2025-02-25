import pytest
from unittest.mock import patch, Mock
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete
from src.core.agent import Agent, StateActionRewardPacket
from collections import defaultdict

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


def test_init():
    """
    Test the initialization of the Agent class.
    """
    env = MockEnv(n_actions=4)
    agent = Agent(env, debug=True)
    assert agent._env == env
    assert isinstance(agent._rewards_by_action_state, defaultdict)
    assert agent._rewards_by_action_state.default_factory == int
    assert agent.action is None
    assert agent.ended == False
    assert agent.queue == []
    assert agent.debug == True

def test_set_env():
    """
    Test the set_env method to ensure the environment can be updated.
    """
    env1 = MockEnv(n_actions=4)
    env2 = MockEnv(n_actions=5)
    agent = Agent(env1)
    assert agent._env == env1
    agent.set_env(env2)
    assert agent._env == env2

def test_select_action_greedy():
    """
    Test select_action to ensure it selects the greedy action when epsilon=1.
    """
    env = MockEnv(n_actions=4)
    agent = Agent(env)
    agent._rewards_by_action_state[(0, 2)] = 1
    action = agent.select_action(0, epsilon=1)
    assert action == 2
    assert agent.action == 2

def test_select_action_random():
    """
    Test select_action to ensure it selects a random action when epsilon=0.
    """
    env = MockEnv(n_actions=4)
    agent = Agent(env)
    with patch.object(env.action_space, 'sample', return_value=3):
        action = agent.select_action(0, epsilon=0)
        assert action == 3

def test_select_action_equal_rewards():
    """
    Test select_action to ensure it selects the first action when all rewards are equal.
    """
    env = MockEnv(n_actions=4)
    agent = Agent(env)
    action = agent.select_action(0, epsilon=1)
    assert action == 0

def test_update():
    """
    Test the update method without future rewards.
    """
    env = MockEnv(n_actions=4)
    agent = Agent(env)
    agent._rewards_by_action_state[(0, 0)] = 1
    agent.update(reward=1, observation=0, next_observation=1, action=0, gamma=0.9, epsilon=0.1)
    assert len(agent.queue) == 1
    packet = agent.queue[0]
    assert packet.state == 0
    assert packet.action == 0
    assert np.isclose(packet.reward, 0.9)

def test_update_with_future_reward():
    """
    Test the update method considering future rewards.
    """
    env = MockEnv(n_actions=4)
    agent = Agent(env)
    agent._rewards_by_action_state[(0, 0)] = 1
    agent._rewards_by_action_state[(1, 1)] = 2
    agent.update(reward=1, observation=0, next_observation=1, action=0, gamma=0.9, epsilon=0.1)
    assert len(agent.queue) == 1
    packet = agent.queue[0]
    assert packet.state == 0
    assert packet.action == 0
    assert np.isclose(packet.reward, 1.08)

def test_compute_rewards_win():
    """
    Test compute_rewards for a winning state.
    """
    env = MockEnv(n_actions=4)
    agent = Agent(env)
    packet1 = StateActionRewardPacket(state=0, action=0, reward=1.0)
    packet2 = StateActionRewardPacket(state=1, action=1, reward=2.0)
    agent.queue = [packet1, packet2]
    total_rewards = agent.compute_rewards(win_state=True)
    assert total_rewards == 2.0
    assert agent._rewards_by_action_state[(1, 1)] == 2.0
    assert (0, 0) not in agent._rewards_by_action_state or agent._rewards_by_action_state[(0, 0)] == 0
    assert len(agent.queue) == 0

def test_compute_rewards_lose():
    """
    Test compute_rewards for a losing state.
    """
    env = MockEnv(n_actions=4)
    agent = Agent(env)
    packet1 = StateActionRewardPacket(state=0, action=0, reward=1.0)
    packet2 = StateActionRewardPacket(state=1, action=1, reward=2.0)
    agent.queue = [packet1, packet2]
    total_rewards = agent.compute_rewards(win_state=False)
    assert total_rewards == -2.0
    assert agent._rewards_by_action_state[(1, 1)] == -2.0
    assert len(agent.queue) == 0

def test_clear_queue():
    """
    Test the clear_queue method to ensure the queue is cleared.
    """
    env = MockEnv(n_actions=4)
    agent = Agent(env)
    agent.queue = [1, 2, 3]
    agent.clear_queue()
    assert agent.queue == []

def test_str():
    """
    Test the string representation of the Agent.
    """
    env = MockEnv(n_actions=4)
    agent = Agent(env)
    agent.action = 1
    agent.ended = True
    assert str(agent) == "Action: 1, Ended: True"
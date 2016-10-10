__author__ = 'flyers'

import numpy as np

import gym.spaces
from gym.envs.vrep.vrep_hierarchy_env import VREPHierarchyEnv
from gym.envs.vrep.vrep_hierarchy_target_env import VREPHierarchyTargetEnv

# from arena.games.game import Game
from arena.utils import *
from arena.replay_memory import ReplayMemory

logger = logging.getLogger(__name__)

class VREPHierarchyGame(object):
    def __init__(self, history_length=1, replay_memory_size=1000000, replay_start_size=100,
                 **kwargs):
        self.episode_step = 0
        self.reset_state = None
        self.env = VREPHierarchyEnv(**kwargs)

        self.rng = get_numpy_rng()
        self.history_length = history_length
        if self.env._action_type == 'continuous':
            action_dim = self.env.action_space.shape
        else:
            action_dim = (self.env.action_space.n,)
        self.replay_memory = ReplayMemory(state_dim=self.env.observation_space.shape,
                                          action_dim=action_dim,
                                          state_dtype='float32', action_dtype='float32',
                                          history_length=history_length,
                                          memory_size=replay_memory_size,
                                          replay_start_size=replay_start_size)

    def reset(self):
        self.episode_step = 0
        self.reset_state = self.env.reset()
        return self.reset_state

    def step(self, a):
        self.episode_step += 1
        next_obs, reward, done, info = self.env.step(a)
        self.replay_memory.append(next_obs, a, reward, done)
        return next_obs, reward, done, info

    def current_state(self):
        if self.episode_step == 0:
            obs = np.tile(self.reset_state, (self.history_length,)+tuple(np.ones(self.reset_state.ndim, dtype=np.int8)))
        elif self.replay_memory.size < self.history_length or \
                np.any(self.replay_memory.terminate_flags.take(np.arange(self.replay_memory.top-self.history_length, self.replay_memory.top))):
            top_obs = self.replay_memory.states.take(self.replay_memory.top-1, axis=0, mode='wrap')
            obs = np.tile(top_obs, (self.history_length,)+tuple(np.ones(self.reset_state.ndim, dtype=np.int8)))
        else:
            obs = self.replay_memory.states.take(np.arange(self.replay_memory.top-self.history_length, self.replay_memory.top), axis=0, mode='wrap')
        if len(self.env.observation_space.shape) == 1:
            obs = obs.flatten()
        return obs

    def sample(self, batch_size):
        states, actions, rewards, next_states, terminate_flags \
            = self.replay_memory.sample(batch_size=batch_size)
        if len(self.env.observation_space.shape) == 1:
            states = states.reshape((batch_size, -1))
        return states, actions, rewards, next_states, terminate_flags

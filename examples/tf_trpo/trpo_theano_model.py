from __future__ import absolute_import
from arena.models.model import ModelWithCritic
import numpy as np
import random
import math
import logging
from modular_rl.agentzoo import TrpoAgent

seed = 1
random.seed(seed)
np.random.seed(seed)


class TrpoTheanoModel(ModelWithCritic):
    def __init__(self, observation_space, action_space):
        ModelWithCritic.__init__(self, observation_space, action_space)
        self.ob_space = observation_space
        self.act_space = action_space
        self.cfg = {'timestep_limit': 50, 'timesteps_per_batch': 15000, 'agent': 'modular_rl.agentzoo.TrpoAgent',
                    'metadata': '', 'load_snapshot': '', 'filter': 0, 'snapshot_every': 0, 'plot': False, 'n_iter': 500,
                    'video': 1, 'seed': 0, 'use_hdf': 0, 'outfile': './Reacher', 'lam': 0.97, 'gamma': 0.995,
                    'hid_sizes': [64, 64], 'parallel': 0, 'cg_damping': 0.1, 'env': 'Reacher-v1', 'max_kl': 0.01,
                    'activation': 'tanh'}
        self.agent = TrpoAgent(self.ob_space, self.act_space, self.cfg)
        self.info_shape = dict(prob=(self.act_space.low.shape[0] * 2,))
        self.debug = True

    def predict(self, observation):
        # obs = np.expand_dims(observation, 0)
        action, agent_info = self.agent.act(observation)
        return action, agent_info

    def compute_critic(self, states):
        return self.agent.baseline.predict(states)

    def compute_update(self, sample_data):
        self.agent.baseline.fit(sample_data)
        self.agent.updater(sample_data)

        return None, None

    def update(self, diff, new=None):
        pass
        # self.set_params_with_flat_data(new)

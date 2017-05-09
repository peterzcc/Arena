import numpy as np
import scipy
from scipy import signal
from collections import defaultdict


def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.

    inputs
    ------
    x: ndarray
    gamma: float

    outputs
    -------
    y: ndarray with same shape as x, satisfying

        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    """
    assert x.ndim >= 1
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# TODO: add agent info,important
class DictMemory(object):
    def __init__(self, gamma=0.995, use_gae=False,
                 lam=0.96, normalize=True, timestep_limit=1000):
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae
        self.normalize = normalize
        self.paths = []
        self.current_path = defaultdict(list)
        self.timestep_limit = timestep_limit
        self.f_critic = None

    def append_state(self, observation, action, info):
        self.current_path["observation"].append(observation)
        self.current_path["action"].append(action)
        for (k, v) in info.items():
            self.current_path[k].append(v)

    def append_feedback(self, reward):
        self.current_path["reward"].append(reward)
        # self.current_path["terminated"].append(False)

    def fill_episode_critic(self, f_critic):
        self.f_critic = f_critic
        pass

    def add_path(self, done):
        if done:
            self.current_path["terminated"] = True
        else:
            self.current_path["terminated"] = False
        self.paths.append({k: v for (k, v) in list(self.current_path.items())})
        # self.paths.append({k: np.array(v) for (k, v) in list(self.current_path.items())})
        self.current_path = defaultdict(list)

    def extract_all(self):
        if self.use_gae:
            # Compute return, baseline, advantage
            for path in self.paths:
                path["reward"] = np.array(path["reward"])
                path["action"] = np.array(path["action"])
                path["observation"] = \
                    [np.array([o[0] for o in path["observation"]]),
                     np.array([o[1] for o in path["observation"]]).astype(np.float32) / 255.0]
                path["times"] = np.arange(len(path["reward"])).reshape(-1, 1) / float(self.timestep_limit)
                path["return"] = discount(path["reward"], self.gamma)
                b = path["baseline"] = self.f_critic(path)
                b1 = np.append(b, 0 if path["terminated"] else b[-1])
                deltas = path["reward"] + self.gamma * b1[1:] - b1[:-1]
                path["advantage"] = discount(deltas, self.gamma * self.lam)
        else:
            for path in self.paths:
                path["reward"] = np.array(path["reward"])
                path["action"] = np.array(path["action"])
                path["return"] = discount((path["reward"]), self.gamma)
                b = path["baseline"] = self.f_critic(path)
                path["advantage"] = path["return"] - b
        if self.normalize:
            alladv = np.concatenate([path["advantage"] for path in self.paths])
            # Standardize advantage
            std = alladv.std()
            mean = alladv.mean()
            for path in self.paths:
                path["advantage"] = (path["advantage"] - mean) / std
        return self.paths

    def reset(self):
        self.paths = []
        self.current_path = defaultdict(list)

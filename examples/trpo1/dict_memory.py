import numpy as np
import scipy
from scipy import signal
from collections import defaultdict
import threading as thd
import time
import logging
from read_write_lock import ReadWriteLock
from  functools import reduce
import operator
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
                 lam=0.96, normalize=True, timestep_limit=1000,
                 f_critic=None,
                 num_actors=1,
                 f_check_batch=None):
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae
        self.normalize = normalize
        self.dist_paths = [list() for i in range(num_actors)]
        self.paths = []
        self.num_episodes = 0
        self.current_path = [defaultdict(list) for i in range(num_actors)]
        self.timestep_limit = timestep_limit
        self.f_critic = f_critic
        self.num_actors = num_actors
        self.paths_lock = ReadWriteLock()
        self.global_t = 0
        self.num_epoch = 0
        self.time_count = 0  # np.zeros((num_actors,),dtype=int)
        self.f_check_batch = f_check_batch
        self.run_start_time = None

    def append_state(self, observation, action, info, pid=0):
        if self.run_start_time is None:
            self.run_start_time = time.time()
        self.current_path[pid]["observation"].append(observation)
        self.current_path[pid]["action"].append(action)
        for (k, v) in info.items():
            self.current_path[pid][k].append(v)

    def append_feedback(self, reward, pid=0):
        self.current_path[pid]["reward"].append(reward)
        # self.current_path["terminated"].append(False)

    # def fill_episode_critic(self, f_critic):
    #     self.f_critic = f_critic
    #     pass

    def add_path(self, done, pid=0):
        if done:
            self.current_path[pid]["terminated"] = True
        else:
            self.current_path[pid]["terminated"] = False

        self.paths_lock.acquire_write()
        self.dist_paths[pid].append({k: v for (k, v) in list(self.current_path[pid].items())})
        episode_len = len(self.current_path[pid]["action"])
        self.time_count += episode_len
        self.global_t += episode_len
        time_count = self.time_count
        self.num_episodes += 1
        episode_count = self.num_episodes
        if self.f_check_batch(time_count, episode_count):
            run_end_time = time.time()
            paths = self.extract_all()

            self.paths_lock.release_write()

            extract_end_time = time.time()
            run_time = (run_end_time - self.run_start_time) / time_count
            extract_time = (extract_end_time - run_end_time) / time_count
            self.run_start_time = None
            epoch_reward = np.asscalar(np.sum(np.concatenate([p["reward"] for p in paths])))
            self.num_epoch += 1
            logging.info(
                'Epoch:%d \nt: %d\nAverage Return:%f, \nNum steps: %d\nNum traj:%d\nte:%f\nt_ex:%f\n' \
                % (self.num_epoch,
                   self.global_t,
                   epoch_reward / episode_count,
                   time_count,
                   episode_count,
                   run_time,
                   extract_time
                   ))
            result = {"paths": paths, "time_count": time_count}
        else:
            self.paths_lock.release_write()
            result = None
        # self.paths.append({k: np.array(v) for (k, v) in list(self.current_path.items())})
        self.current_path[pid] = defaultdict(list)

        return result

    def extract_all(self):
        self.paths = reduce(operator.add, self.dist_paths)
        self.dist_paths = [[] for i in range(self.num_actors)]
        if self.use_gae:
            # Compute return, baseline, advantage
            for path in self.paths:
                path["reward"] = np.array(path["reward"])
                path["action"] = np.array(path["action"])
                if len(path["observation"][0]) == 2:
                    path["observation"] = \
                        [np.array([o[0] for o in path["observation"]]),
                         np.array([o[1] for o in path["observation"]]).astype(np.float32) / 255.0]
                else:
                    path["observation"] = \
                        [np.array([o[0] for o in path["observation"]])]
                path["times"] = np.arange(len(path["reward"])).reshape(-1, 1) / float(self.timestep_limit)
                if self.gamma < 0.999:  # don't scale for gamma ~= 1
                    scaled_rewards = path['reward'] * (1 - self.gamma)
                else:
                    scaled_rewards = path['reward']
                path["return"] = discount(scaled_rewards, self.gamma)
                b = path["baseline"] = self.f_critic(path)
                b1 = np.append(b, 0 if path["terminated"] else b[-1])
                deltas = scaled_rewards + self.gamma * b1[1:] - b1[:-1]
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
                path["advantage"] = (path["advantage"] - mean) / (std + 1e-6)
        paths = self.paths.copy()
        self.paths = []
        self.time_count = 0
        self.num_episodes = 0
        return paths

    def reset(self):
        with self.paths_lock:
            self.paths = []
        self.time_count = 0
        self.current_path = [defaultdict(list) for i in range(self.num_actors)]
        self.num_episodes = 0
        self.dist_paths = [list() for i in range(self.num_actors)]

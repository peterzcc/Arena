import numpy as np
import scipy
from scipy import signal
from collections import defaultdict
import threading as thd
import time
import logging
from read_write_lock import ReadWriteLock
from multiprocessing import Lock
from  functools import reduce
import operator
from arena.experiment import Experiment
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


class DictMemory(object):
    def __init__(self, gamma=0.995,
                 lam=0.96, normalize=True, timestep_limit=1000,
                 f_critic=None,
                 num_actors=1,
                 f_check_batch=None,
                 async=False):
        self.gamma = gamma
        self.lam = lam
        self.normalize = normalize
        self.dist_paths = [list() for i in range(num_actors)]

        self.current_path = [defaultdict(list) for i in range(num_actors)]
        self.timestep_limit = timestep_limit
        self.f_critic = f_critic
        self.num_actors = num_actors
        self.paths_lock = Lock()
        self.global_t = 0
        self.num_epoch = 0
        self.time_count = 0
        # self.time_counts = np.zeros((num_actors,), dtype=np.int)
        self.num_episodes = 0
        # self.pid_n_episodes = np.zeros((num_actors,), dtype=np.int)

        self.f_check_batch = f_check_batch
        self.run_start_time = None
        # self.run_start_times = np.zeros((num_actors,), dtype=np.float32)

        self.gpow = self.gamma ** np.arange(0, timestep_limit)

    def append_state(self, observation, action, info, pid=0):
        if self.run_start_time is None:
            self.run_start_time = time.time()
        self.current_path[pid]["observation"].append(observation)
        self.current_path[pid]["action"].append(action)
        for (k, v) in info.items():
            self.current_path[pid][k].append(v)

    def append_observation(self, observation, pid=0):
        if self.run_start_time is None:
            self.run_start_time = time.time()
        self.current_path[pid]["observation"].append(observation)

    def append_action(self, action, info, pid=0):
        self.current_path[pid]["action"].append(action)
        for (k, v) in info.items():
            self.current_path[pid][k].append(v)
        self.current_path[pid]["pos"].append(len(self.current_path[pid]["reward"]))

    def append_feedback(self, reward, pid=0):
        self.current_path[pid]["reward"].append(reward)
        # self.time_counts[pid] += 1
        self.time_count += 1

    def incre_count_and_check_done(self, pid=None):
        if pid is None:
            batch_ends = self.f_check_batch(self.time_count, self.num_episodes)
        else:
            batch_ends = self.f_check_batch(self.time_counts[pid], self.pid_n_episodes[pid])

        return batch_ends

    def profile_extract_all(self, pid=None):
        if pid is None:
            time_count = self.time_count
            episode_count = self.num_episodes
            run_end_time = time.time()
            paths = self.extract_all(pid)
            extract_end_time = time.time()
            run_time = (run_end_time - self.run_start_time) / time_count
            extract_time = (extract_end_time - run_end_time) / time_count
            self.run_start_time = None
            self.num_epoch += 1
            logging.info("Name: {}".format(Experiment.EXP_NAME))
            logging.info(
                'Epoch:%d \nt: %d\nNum steps: %d\nNum traj:%d\nte:%f\nt_ex:%f\n' \
                % (self.num_epoch,
                   self.global_t,
                   time_count,
                   episode_count,
                   run_time,
                   extract_time
                   ))
            result = {"paths": paths, "time_count": time_count}
        else:
            result = {}
            # time_count = self.time_counts[pid]
            # episode_count = self.pid_n_episodes[pid]
            # run_end_time = time.time()
            # paths = self.extract_all(pid)
            # extract_end_time = time.time()
            # run_time = (run_end_time - self.run_start_times[pid]) / time_count
            # extract_time = (extract_end_time - run_end_time) / time_count
            # self.run_start_times[pid] = 0
            # with self.paths_lock:
            #     self.num_epoch += 1
            # if pid == 0:
            #     logging.info("Name: {}".format(Experiment.EXP_NAME))
            #     logging.info(
            #         'Epoch:%d \nt: %d\nNum steps: %d\nNum traj:%d\nte:%f\nt_ex:%f\n' \
            #         % (self.num_epoch,
            #            self.global_t,
            #            time_count,
            #            episode_count,
            #            run_time,
            #            extract_time
            #            ))
            # result = {"paths": paths, "time_count": time_count}

        return result

    def transfer_single_path(self, done, pid=0):
        if done:
            self.current_path[pid]["terminated"] = True
        else:
            self.current_path[pid]["terminated"] = False
        self.dist_paths[pid].append({k: v for (k, v) in list(self.current_path[pid].items())})
        episode_len = len(self.current_path[pid]["action"])

        with self.paths_lock:
            self.global_t += episode_len
            self.num_episodes += 1
        # self.pid_n_episodes[pid] += 1
        self.current_path[pid] = defaultdict(list)

    def extract_paths(self, pid=None):
        if pid is None:
            with self.paths_lock:
                result_paths = reduce(operator.add, self.dist_paths)
                self.dist_paths = [[] for i in range(self.num_actors)]
                self.time_count = 0
                self.num_episodes = 0
            # self.time_counts = np.zeros((self.num_actors,), dtype=np.int)
            # self.pid_n_episodes = np.zeros((self.num_actors,), dtype=np.int)
            return result_paths
        else:
            result_paths = self.dist_paths[pid].copy()
            self.dist_paths[pid] = []
            # with self.paths_lock:
            #     self.time_count -= self.time_counts[pid]
            #     self.num_episodes -= self.pid_n_episodes[pid]
            # self.time_counts[pid] = 0
            # self.pid_n_episodes[pid] = 0
            return result_paths

    def extract_all(self, pid=None):
        paths = self.extract_paths(pid)
        for path in paths:
            path["observation"] = \
                [np.array([o[i] for o in path["observation"]]) for i in range(len(path["observation"][0]))]
            # if len(path["observation"][0]) >= 2:
            #     path["observation"] = \
            #         [np.array([o[i] for o in path["observation"]]) for i in range(len(path["observation"][0]))]
            # else:
            #     path["observation"] = \
            #         [np.array([o[0] for o in path["observation"]])]
        if "pos" in paths[0]:
            for path in paths:
                path["reward"] = np.array(path["reward"])
                path["action"] = np.array(path["action"])

                path["times"] = \
                    (np.arange(len(path["reward"])).reshape(-1, 1) / float(self.timestep_limit))[path["pos"], :]

                # semi_mdp_r = np.zeros(path["action"].shape)

                grouped_rewards = np.split(path["reward"], path["pos"][1:])
                group_lens = list(map(len, grouped_rewards))
                grouped_gpow = [self.gpow[0:glen] for glen in group_lens]
                semi_mdp_r = np.array([np.inner(r, p) for r, p in zip(grouped_rewards, grouped_gpow)])


                path["return"] = discount(semi_mdp_r, self.gamma)
                b = path["baseline"] = self.f_critic(path)
                b1 = np.append(b, 0 if path["terminated"] else b[-1])

                deltas = semi_mdp_r + self.gamma * b1[1:] - b1[:-1]
                path["advantage"] = discount(deltas, self.gamma * self.lam)
        else:
            for path in paths:
                path["reward"] = np.array(path["reward"])
                path["action"] = np.array(path["action"])
                path["times"] = np.arange(len(path["reward"])).reshape(-1, 1) / float(self.timestep_limit)
                path["return"] = discount(path['reward'], self.gamma)
                b = path["baseline"] = self.f_critic(path)
                b1 = np.append(b, 0 if path["terminated"] else b[-1])

                deltas = path['reward'] + self.gamma * b1[1:] - b1[:-1]
                path["advantage"] = discount(deltas, self.gamma * self.lam)
        if self.normalize:
            alladv = np.concatenate([path["advantage"] for path in paths])
            # Standardize advantage
            std = alladv.std()
            mean = alladv.mean()
            for path in paths:
                path["advantage"] = (path["advantage"]-mean) / (std + 1e-6)
        return paths

        # def reset(self):
        #     self.time_count = 0
        #     self.current_path = [defaultdict(list) for i in range(self.num_actors)]
        #     self.num_episodes = 0
        #     self.dist_paths = [list() for i in range(self.num_actors)]

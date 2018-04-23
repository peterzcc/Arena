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
from itertools import compress
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
                 f_critic_root=None,
                 num_leafs=0,
                 f_critic_leafs=[],
                 num_actors=1,
                 f_check_batch=None):
        self.gamma = gamma
        self.lam = lam
        self.normalize = normalize
        self.dist_paths = [list() for i in range(num_actors)]

        self.current_path = [defaultdict(list) for i in range(num_actors)]
        self.timestep_limit = timestep_limit
        self.f_critic = f_critic_root
        self.num_leafs = num_leafs
        self.f_critic_leafs = f_critic_leafs
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

    def append_state(self, observation, action, info, pid=0,
                     leaf_id=None, leaf_action=None, leaf_model_info=None, curr_time_step=None):
        if self.run_start_time is None:
            self.run_start_time = time.time()
        self.current_path[pid]["observation"].append(observation)
        self.current_path[pid]["action"].append(action)
        self.current_path[pid]["time_unormalized"].append(curr_time_step)
        for (k, v) in info.items():
            self.current_path[pid]["root_" + k].append(v)
        if leaf_id is not None:
            self.current_path[pid]["leaf_id"].append(leaf_id)
            self.current_path[pid]["leaf_action"].append(leaf_action)
            for k, v in leaf_model_info.items():
                full_key = "leaf_" + k
                self.current_path[pid][full_key].append(v)

    def append_observation(self, observation, pid=0):
        if self.run_start_time is None:
            self.run_start_time = time.time()
        self.current_path[pid]["observation"].append(observation)

    def append_action(self, action, info, pid=0):
        self.current_path[pid]["action"].append(action)
        for (k, v) in info.items():
            self.current_path[pid][k].append(v)
        self.current_path[pid]["pos"].append(len(self.current_path[pid]["reward"]))

    def append_feedback(self, reward, pid=0, info={}):
        self.current_path[pid]["reward"].append(reward)
        for k, v in info.items():
            self.current_path[pid][k].append(v)
        # self.time_counts[pid] += 1
        self.time_count += 1

    def incre_count_and_check_done(self, pid=None):
        if pid is None:
            batch_ends = self.f_check_batch(self.time_count, self.num_episodes)
        else:
            # batch_ends = self.f_check_batch(self.time_counts[pid], self.pid_n_episodes[pid])
            raise NotImplementedError

        return batch_ends

    def profile_extract_all(self, pid=None, with_subtasks=False):
        if pid is None:
            time_count = self.time_count
            episode_count = self.num_episodes
            run_end_time = time.time()
            extracted_result = self.extract_all(pid, with_subtasks=with_subtasks)
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
            result = {**extracted_result, "time_count": time_count}
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

    def stack_obs_for_paths(self, paths):
        obs = []
        for i in range(len(paths[0]["observation"][0])):
            obs.append(np.concatenate([[o[i] for o in path["observation"]] for path in paths]))

            # for path in paths:
            #     path["observation"] = \
            #         [np.array([o[i] for o in path["observation"]]) for i in range(len(path["observation"][0]))]
            # if len(path["observation"][0]) >= 2:
            #     path["observation"] = \
            #         [np.array([o[i] for o in path["observation"]]) for i in range(len(path["observation"][0]))]
            # else:
            #     path["observation"] = \
            #         [np.array([o[0] for o in path["observation"]])
        return obs

    def append_term(self, b, ter):
        return np.append(b, 0 if ter else b[-1])

    def compute_delta(self, reward, b1):
        return reward + self.gamma * b1[1:] - b1[:-1]

    def compute_gae_for_path(self, path, rewards=None, f_critic=None):

        if rewards is None:
            rewards = np.split(path['reward'], path["splits"])
        returns = list(map(lambda x: discount(x, self.gamma),
                           rewards))
        # path["return"] = discount(path['reward'], self.gamma)
        path["baseline"] = f_critic(path)
        b = np.split(path["baseline"], path["splits"])

        # b1 = np.append(b, 0 if path["terminated"] else b[-1])
        b1 = list(map(self.append_term, b, path["terminated"]))

        # deltas = path['reward'] + self.gamma * b1[1:] - b1[:-1]
        deltas = list(map(self.compute_delta, rewards, b1))
        advs = list(map(lambda x: discount(x, self.gamma * self.lam), deltas))
        # path["advantage"] = discount(deltas, self.gamma * self.lam)
        path["advantage"] = np.concatenate(advs)
        path["return"] = np.concatenate(returns)

    def normalize_gae(self, paths):
        alladv = paths["advantage"]  # np.concatenate([path["advantage"] for path in paths])
        # Standardize advantage
        std = alladv.std()
        mean = alladv.mean()

        # for path in paths:
        #     path["advantage"] = (path["advantage"] - mean) / (std + 1e-6)
        paths["advantage"] = (paths["advantage"] - mean) / (std + 1e-6)

    def extract_all(self, pid=None, with_subtasks=False):
        paths = self.extract_paths(pid)
        obs = self.stack_obs_for_paths(paths)
        root_paths = {}
        root_paths["observation"] = obs

        # if "pos" in paths[0]:
        #     for path in paths:
        #         path["reward"] = np.array(path["reward"])
        #         path["action"] = np.array(path["action"])
        #
        #         path["times"] = \
        #             (np.arange(len(path["reward"])).reshape(-1, 1) / float(self.timestep_limit))[path["pos"], :]
        #
        #         # semi_mdp_r = np.zeros(path["action"].shape)
        #
        #         grouped_rewards = np.split(path["reward"], path["pos"][1:])
        #         group_lens = list(map(len, grouped_rewards))
        #         grouped_gpow = [self.gpow[0:glen] for glen in group_lens]
        #         semi_mdp_r = np.array([np.inner(r, p) for r, p in zip(grouped_rewards, grouped_gpow)])
        #
        #
        #         path["return"] = discount(semi_mdp_r, self.gamma)
        #         b = path["baseline"] = self.f_critic(path)
        #         b1 = np.append(b, 0 if path["terminated"] else b[-1])
        #
        #         deltas = semi_mdp_r + self.gamma * b1[1:] - b1[:-1]
        #         path["advantage"] = discount(deltas, self.gamma * self.lam)
        path_lens = [len(p["reward"]) for p in paths]
        root_paths["splits"] = np.cumsum(path_lens, dtype=int)
        root_paths["terminated"] = np.array([p["terminated"] for p in paths])
        for k in ["action", "reward", "time_unormalized"]:
            root_paths[k] = np.concatenate([p[k] for p in paths])
        root_paths["times"] = root_paths["time_unormalized"].reshape(-1, 1) / float(self.timestep_limit)
        root_prefix = "root_"
        for k, _ in paths[0].items():
            if k.startswith(root_prefix):
                root_paths[k[len(root_prefix):]] = np.concatenate([p[k] for p in paths])
        # for path in paths:
        #     path["reward"] = np.array(path["reward"])
        #     path["action"] = np.array(path["action"])
        #     path["times"] =
        self.compute_gae_for_path(root_paths, f_critic=self.f_critic)

        if self.normalize:
            self.normalize_gae(root_paths)
        result = {"paths": root_paths}
        if with_subtasks:
            agg_paths = {}
            leaf_prefix = "leaf_"
            for k in paths[0].keys():
                if k.startswith(leaf_prefix):
                    agg_paths[k[len(leaf_prefix):]] = np.concatenate([p[k] for p in paths])
            for k in ["observation", "times"]:
                agg_paths[k] = root_paths[k]
            subrewards = np.concatenate([p["subrewards"] for p in paths])
            agg_paths["reward"] = np.choose(agg_paths["id"], subrewards.T)

            is_leaf_split = np.zeros(root_paths["reward"].shape, dtype=np.bool)
            leaf_term = np.zeros(root_paths["reward"].shape, dtype=np.bool)

            is_leaf_split[root_paths["splits"] - 1] = True
            leaf_term[root_paths["splits"] - 1] = root_paths["terminated"]

            leaf_exits = np.flatnonzero(root_paths["action"])
            is_leaf_split[leaf_exits] = True
            is_leaf_split[0] = False

            full_leaf_splits = np.append(np.flatnonzero(is_leaf_split), is_leaf_split.shape[0])
            full_leaf_terms = leaf_term[full_leaf_splits - 1]
            full_leaf_ids = agg_paths["id"][full_leaf_splits - 1]
            del agg_paths["id"]

            splitted_paths = {}
            for k in agg_paths.keys():
                if k == "observation":
                    splitted_paths[k] = [None] * len(agg_paths[k])
                    for i in range(len(agg_paths[k])):
                        splitted_paths[k][i] = np.split(agg_paths[k][i], full_leaf_splits)
                else:
                    splitted_paths[k] = np.split(agg_paths[k], full_leaf_splits)

            leaf_paths = [dict() for i in range(self.num_leafs)]
            for l in range(self.num_leafs):
                this_paths = {}
                is_this_leaf = full_leaf_ids == l
                num_data = np.count_nonzero(is_this_leaf)
                if num_data > 0:
                    for k in splitted_paths.keys():
                        if k == "observation":
                            this_paths[k] = [None] * len(splitted_paths[k])
                            leaf_paths[l][k] = [None] * len(splitted_paths[k])
                            for i in range(len(splitted_paths[k])):
                                this_paths[k][i] = list(compress(splitted_paths[k][i], is_this_leaf))
                                leaf_paths[l][k][i] = np.concatenate(this_paths[k][i])
                        else:
                            this_paths[k] = list(compress(splitted_paths[k], is_this_leaf))
                            leaf_paths[l][k] = np.concatenate(this_paths[k])
                    lens = list(map(len, this_paths["times"]))
                    leaf_paths[l]["splits"] = np.cumsum(lens)
                    leaf_paths[l]["terminated"] = full_leaf_terms[is_this_leaf]
                    self.compute_gae_for_path(leaf_paths[l], rewards=this_paths["reward"],
                                              f_critic=self.f_critic_leafs[l])
                    if self.normalize:
                        self.normalize_gae(leaf_paths[l])
                else:
                    leaf_paths[l] = None
            result["leaf_paths"] = leaf_paths
        return result

        # def reset(self):
        #     self.time_count = 0
        #     self.current_path = [defaultdict(list) for i in range(self.num_actors)]
        #     self.num_episodes = 0
        #     self.dist_paths = [list() for i in range(self.num_actors)]

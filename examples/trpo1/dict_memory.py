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
    # assert x.ndim >= 1
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def var_discount(x, t, pow_series):
    discounts = pow_series[t]
    return np.sum(x * discounts)


def valid_split(x, s):
    return rm_empty(np.split(x, s))

def reverse_cumsum(x):
    return np.cumsum(x[::-1])[::-1]

def rm_empty(l):
    if l[0].size == 0:
        return l[1:]
    elif l[-1].size == 0:
        return l[:-1]
    else:
        return l

SWITCHER_PREFIX = "switcher_"
SWITCHER_ACTION = "switcher_action"
DECIDER_PREFIX = "decider_"
DECIDER_ACTION = "decider_action"
OBSERVATION = "observation"
TIMES = "times"
REWARD = "reward"
TERMINATED = "terminated"
HAS_DECISION = "has_decision"
SPLITS = "splits"


class DictMemory(object):
    def __init__(self, gamma=0.995,
                 lam=0.96, normalize=True, timestep_limit=1000,
                 f_critic={},
                 num_leafs=0,
                 num_actors=1,
                 f_check_batch=None):
        self.gamma = gamma
        self.lam = lam
        self.normalize = normalize
        self.dist_paths = [list() for i in range(num_actors)]

        self.current_path = [defaultdict(list) for i in range(num_actors)]
        self.timestep_limit = timestep_limit
        self.f_critic = f_critic
        self.num_leafs = num_leafs
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

        self.gamma_pow = np.power(self.gamma, np.arange(0, timestep_limit), dtype=np.float64)
        self.lam_pow = np.power(self.lam, np.arange(0, timestep_limit), dtype=np.float64)

        self.leaf_info_keys = None

    def append_state(self, observation, action, info, pid=0,
                     leaf_id=None, leaf_action=None, leaf_model_info=None, curr_time_step=None):
        if self.run_start_time is None:
            self.run_start_time = time.time()
        self.current_path[pid][OBSERVATION].append(observation)
        self.current_path[pid]["action"].append(action)
        self.current_path[pid]["time_unormalized"].append(curr_time_step)
        for (k, v) in info.items():
            self.current_path[pid]["root_" + k].append(v)
        if leaf_id is not None:
            self.current_path[pid]["leaf_id"].append(leaf_id)
            self.current_path[pid]["leaf_{}_".format(leaf_id) + "action"].append(leaf_action)
            if self.leaf_info_keys is None:
                self.leaf_info_keys = leaf_model_info.keys()
            for k, v in leaf_model_info.items():
                full_key = "leaf_{}_{}".format(leaf_id, k)
                self.current_path[pid][full_key].append(v)

    def append_hrl_state(self, observation, should_switch, switcher_model_info, decision, decider_model_info, pid=0,
                         leaf_id=None, leaf_action=None, leaf_model_info=None, curr_time_step=None):
        if self.run_start_time is None:
            self.run_start_time = time.time()
        self.current_path[pid]["time_unormalized"].append(curr_time_step)
        self.current_path[pid][OBSERVATION].append(observation)
        self.current_path[pid][SWITCHER_ACTION].append(should_switch)
        for (k, v) in switcher_model_info.items():
            self.current_path[pid][SWITCHER_PREFIX + k].append(v)
        has_decision = decision is not None
        self.current_path[pid][HAS_DECISION].append(has_decision)
        if has_decision:
            self.current_path[pid][DECIDER_ACTION].append(decision)
            for (k, v) in decider_model_info.items():
                self.current_path[pid][DECIDER_PREFIX + k].append(v)
        if leaf_id is not None:
            self.current_path[pid]["leaf_id"].append(leaf_id)
            self.current_path[pid]["leaf_{}_".format(leaf_id) + "action"].append(leaf_action)
            if self.leaf_info_keys is None:
                self.leaf_info_keys = leaf_model_info.keys()
            for k, v in leaf_model_info.items():
                full_key = "leaf_{}_{}".format(leaf_id, k)
                self.current_path[pid][full_key].append(v)

    def append_observation(self, observation, pid=0):
        if self.run_start_time is None:
            self.run_start_time = time.time()
        self.current_path[pid][OBSERVATION].append(observation)

    def append_action(self, action, info, pid=0):
        self.current_path[pid]["action"].append(action)
        for (k, v) in info.items():
            self.current_path[pid][k].append(v)
        self.current_path[pid]["pos"].append(len(self.current_path[pid][REWARD]))

    def append_feedback(self, reward, pid=0, info={}):
        self.current_path[pid][REWARD].append(reward)
        for k, v in info.items():
            self.current_path[pid][k].append(v)
        # self.time_counts[pid] += 1
        self.time_count += 1

    def check_done(self, pid=None):
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
            self.current_path[pid][TERMINATED] = True
        else:
            self.current_path[pid][TERMINATED] = False
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
        for i in range(len(paths[0][OBSERVATION][0])):
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

    def compute_var_delta(self, reward, b1, gamma_diff):
        return reward + gamma_diff * b1[1:] - b1[:-1]

    def compute_return(self, reward, b, ter):
        bootstraped_reward = reward.copy()
        if not ter:
            bootstraped_reward[-1] = b[-1]
        return discount(bootstraped_reward, self.gamma)

    def compute_smdp_return(self, reward, b, ter, gamma_powered):
        bootstraped_reward = reward.copy()
        if not ter:
            bootstraped_reward[-1] = b[-1]
        return_scaled = reverse_cumsum(bootstraped_reward * gamma_powered) / gamma_powered
        return return_scaled

    def compute_gae_for_path(self, path, rewards=None, f_critic=None, normalize=False):
        results = {}

        if rewards is None:
            rewards = valid_split(path['reward'], path[SPLITS])

        results["baseline"] = f_critic(path)
        b = valid_split(results["baseline"], path[SPLITS])

        # returns = list(map(lambda x: discount(x, self.gamma),
        #                    rewards))
        returns = list(map(self.compute_return, rewards, b, path[TERMINATED]))
        whole_returns = np.concatenate(returns)
        results["return"] = whole_returns

        b1 = list(map(self.append_term, b, path[TERMINATED]))

        # deltas = path['reward'] + self.gamma * b1[1:] - b1[:-1]
        deltas = list(map(self.compute_delta, rewards, b1))
        # advs = list(map(lambda x: discount(x, self.gamma * self.lam), deltas))
        # correctly scale the advs
        lens = [len(r) for r in rewards]
        decision_times = list(map(lambda l: np.arange(0, l), lens))
        whole_decision_times = np.concatenate(decision_times)
        whole_gamma_powered = self.gamma_pow[whole_decision_times]
        gamma_powered = valid_split(whole_gamma_powered, path[SPLITS])

        # whole_lam_powered = self.lam_pow[whole_decision_times]
        # lam_powered = valid_split(whole_lam_powered, path[SPLITS])
        # lam_discounted_accum = list(map(lambda t: reverse_cumsum(t), lam_powered))
        lam_discounted_accum = list(
            map(lambda l: discount(np.ones(l, dtype=np.float64), self.lam), lens))
        advs = list(
            map(lambda d, l: discount(d * l, self.gamma * self.lam) / l,
                deltas, lam_discounted_accum)
        )
        results["advantage"] = np.concatenate(advs)


        if normalize:
            self.normalize_gae(results)
        return results

    def time_diff_with_zero_padding(self, pow_series, x):
        if x.size == 1:
            return np.array(pow_series[0])
        else:
            return np.append(pow_series[x[1:] - x[:-1]], 0)

    def gae_hrl(self, path, rewards=None, f_critic=None, normalize=False):
        results = {}

        if rewards is None:
            rewards = valid_split(path[REWARD], path[SPLITS])
        whole_decision_times = path["decision_times"]
        decision_times = valid_split(whole_decision_times, path[SPLITS])
        whole_gamma_powered = self.gamma_pow[whole_decision_times]
        whole_lam_powered = self.lam_pow[whole_decision_times]
        gamma_powered = valid_split(whole_gamma_powered, path[SPLITS])
        lam_powered = valid_split(whole_lam_powered, path[SPLITS])

        results["baseline"] = f_critic(path)
        b = valid_split(results["baseline"], path[SPLITS])

        b1 = list(map(self.append_term, b, path[TERMINATED]))

        gamma_diffs = list(map(lambda x: self.time_diff_with_zero_padding(self.gamma_pow, x),
                               decision_times))
        # lam_diffs = list(map(lambda x: self.time_diff_with_zero_padding(self.lam_pow, x),
        #                        decision_times))
        # returns_scaled = list(
        #     map(lambda r, g: reverse_cumsum(r * g)/g, rewards, gamma_powered)
        # )

        returns_scaled = list(map(
            self.compute_smdp_return, rewards, b, path[TERMINATED], gamma_powered
        ))

        whole_returns_scaled = np.concatenate(returns_scaled)
        results["return"] = whole_returns_scaled

        deltas = list(
            map(lambda r, b, g: self.compute_var_delta(r, b, g),
                rewards, b1, gamma_diffs))
        lam_discounted_accum = list(map(lambda t: reverse_cumsum(t), lam_powered))
        whole_lam_discounted_accum = np.concatenate(lam_discounted_accum)

        adv_scaled = list(
            map(lambda d, g, l: reverse_cumsum(d * g * l) / (g * l),
                deltas, gamma_powered, lam_discounted_accum))
        # whole_adv_unscaled = np.concatenate(adv_unscaled)
        # whole_adv_scaled = whole_adv_unscaled / (whole_lam_discounted_accum * whole_gamma_powered)
        whole_adv_scaled = np.concatenate(adv_scaled)
        results["advantage"] = whole_adv_scaled

        if normalize:
            self.normalize_gae(results)
        return results

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
        common_data = {}
        common_data[OBSERVATION] = obs

        path_lens = [len(p[REWARD]) for p in paths]
        # common_data["decision_times"] = np.concatenate(list(map(lambda l: np.arange(0, l), path_lens)))
        common_data[SPLITS] = np.cumsum(path_lens, dtype=int)
        common_data[TERMINATED] = np.array([p[TERMINATED] for p in paths])
        for k in [REWARD, "time_unormalized"]:
            common_data[k] = np.concatenate([p[k] for p in paths])
        common_data[TIMES] = common_data["time_unormalized"].reshape(-1, 1) / float(self.timestep_limit)
        if not with_subtasks:
            for k in ["action", ]:
                common_data[k] = np.concatenate([p[k] for p in paths])
            root_prefix = "root_"
            for k in paths[0].keys():
                if k.startswith(root_prefix):
                    common_data[k[len(root_prefix):]] = np.concatenate([p[k] for p in paths])
            gae_infos = self.compute_gae_for_path(common_data, f_critic=self.f_critic["decider"],
                                                  normalize=self.normalize)
            common_data.update(**gae_infos)
            result = {"paths": common_data}
        else:
            for k in [HAS_DECISION, ]:
                common_data[k] = np.concatenate([p[k] for p in paths])
            result = {}
            # switch policy
            switcher_data = {}
            switcher_data.update(**common_data)
            for k in paths[0].keys():
                if k.startswith(SWITCHER_PREFIX):
                    switcher_data[k[len(SWITCHER_PREFIX):]] = np.concatenate([p[k] for p in paths])
            switcher_gae_info = self.compute_gae_for_path(switcher_data, f_critic=self.f_critic["switcher"],
                                                          normalize=self.normalize)
            switcher_data.update(switcher_gae_info)
            result.update(switcher_data=switcher_data)

            agg_paths = {}
            leaf_data = [dict() for i in range(self.num_leafs)]
            leaf_prefix = "leaf_"
            for k in ["leaf_id"]:
                if k.startswith(leaf_prefix):
                    agg_paths[k[len(leaf_prefix):]] = np.concatenate([p[k] for p in paths])
            for pi in range(self.num_leafs):
                leaf_i_prefix = "leaf_{}_".format(pi)
                for k in list(self.leaf_info_keys) + ["action"]:
                    full_k = leaf_i_prefix + k
                    list_k = [p[full_k] for p in paths if full_k in p]
                    if len(list_k) != 0:
                        leaf_data[pi][k] = np.concatenate(list_k)
                    else:
                        leaf_data[pi][k] = None
            for k in [OBSERVATION, TIMES]:
                agg_paths[k] = common_data[k]
            if "subrewards" in paths[0]:
                train_sub = True
                subrewards = np.concatenate([p["subrewards"] for p in paths])
                agg_paths[REWARD] = np.choose(agg_paths["id"], subrewards.T)
            else:
                train_sub = False
            # agg_paths["index"] = np.arange(0, agg_paths[TIMES].shape[0])

            is_leaf_split = np.zeros(common_data[REWARD].shape, dtype=np.bool)
            leaf_term = np.zeros(common_data[REWARD].shape, dtype=np.bool)

            is_leaf_split[common_data[SPLITS][:-1]] = True
            leaf_term[common_data[SPLITS] - 1] = common_data[TERMINATED]

            leaf_exits = np.flatnonzero(common_data[HAS_DECISION])
            is_leaf_split[leaf_exits] = True
            is_leaf_split[0] = False

            full_leaf_splits = np.append(np.flatnonzero(is_leaf_split), is_leaf_split.shape[0])
            full_leaf_terms = leaf_term[full_leaf_splits - 1]
            full_leaf_ids = agg_paths["id"][full_leaf_splits - 1]
            decider_data = {}
            for k in paths[0].keys():
                if k.startswith(DECIDER_PREFIX):
                    decider_data[k[len(DECIDER_PREFIX):]] = np.concatenate([p[k] for p in paths if k in p])
            for k in [OBSERVATION, ]:
                decider_data[k] = [obs[leaf_exits] for obs in common_data[k]]
            for k in [TIMES, ]:
                decider_data[k] = common_data[k][leaf_exits]
            for k in [TERMINATED]:
                decider_data[k] = common_data[k]

            # decider path

            decider_grouped_rewards = valid_split(common_data[REWARD], leaf_exits)
            group_lens = list(map(len, decider_grouped_rewards))
            grouped_gpow = list(
                map(lambda l: self.gamma_pow[0:l], group_lens))  # [self.gamma_pow[0:glen] for glen in group_lens]
            smdp_rs = list(map(np.inner, decider_grouped_rewards, grouped_gpow))
            smdp_r = np.array(smdp_rs)
            decider_data[REWARD] = smdp_r
            smdp_l = np.array(list(map(np.size, decider_grouped_rewards)))
            decider_data["sublength"] = smdp_l
            decision_times = list(map(np.flatnonzero, [np.array(path[HAS_DECISION]) for path in paths]))
            decider_data["decision_times"] = np.concatenate(decision_times)
            # decider_data["return"] = switcher_data["return"][leaf_exits]
            assert smdp_r.size == decider_data[TIMES].size

            # split
            full_is_decider_split = np.zeros(common_data[REWARD].shape, dtype=np.bool)
            full_is_decider_split[common_data[SPLITS][:-1] - 1] = True
            # grouped_is_split = valid_split(full_is_decider_split, full_leaf_splits)
            # is_decider_split = np.array(list(map(lambda x: x[-1], grouped_is_split)))
            is_decider_split = full_is_decider_split[full_leaf_splits - 1]
            is_decider_split[-1] = False
            decider_data[SPLITS] = np.flatnonzero(is_decider_split) + 1
            assert len(decider_data[SPLITS]) == len(decider_data[TERMINATED]) - 1
            # zero_critic = lambda x: np.zeros(shape=x[REWARD].shape)
            decider_gae = self.gae_hrl(decider_data, f_critic=self.f_critic["decider"],
                                       normalize=self.normalize)
            decider_data.update(**decider_gae)
            result.update(decider_data=decider_data)

            # flat_gae_info = self.compute_gae_for_path(switcher_data, f_critic=self.f_critic["decider"],
            #                                           normalize=self.normalize)

            # del agg_paths["id"]
            splitted_paths = {}
            for k in agg_paths.keys():
                if k == OBSERVATION:
                    splitted_paths[k] = [None] * len(agg_paths[k])
                    for i in range(len(agg_paths[k])):
                        splitted_paths[k][i] = valid_split(agg_paths[k][i], full_leaf_splits)
                else:
                    splitted_paths[k] = valid_split(agg_paths[k], full_leaf_splits)

            for l in range(self.num_leafs):
                this_paths = {}
                is_this_leaf = full_leaf_ids == l
                num_data = np.count_nonzero(is_this_leaf)
                if num_data > 0:
                    for k in splitted_paths.keys():
                        if k == OBSERVATION:
                            this_paths[k] = [None] * len(splitted_paths[k])
                            leaf_data[l][k] = [None] * len(splitted_paths[k])
                            for i in range(len(splitted_paths[k])):
                                this_paths[k][i] = list(compress(splitted_paths[k][i], is_this_leaf))
                                leaf_data[l][k][i] = np.concatenate(this_paths[k][i])
                        else:
                            this_paths[k] = list(compress(splitted_paths[k], is_this_leaf))
                            leaf_data[l][k] = np.concatenate(this_paths[k])
                    lens = list(map(len, this_paths[TIMES]))
                    leaf_data[l][SPLITS] = np.cumsum(lens)
                    leaf_data[l][TERMINATED] = full_leaf_terms[is_this_leaf]
                    if train_sub:
                        results = self.compute_gae_for_path(leaf_data[l], rewards=this_paths[REWARD],
                                                            f_critic=self.f_critic["leafs"][l],
                                                            normalize=self.normalize)
                        leaf_data[l].update(**results)
                else:
                    leaf_data[l] = None
            result["leaf_data"] = leaf_data
        return result

        # def reset(self):
        #     self.time_count = 0
        #     self.current_path = [defaultdict(list) for i in range(self.num_actors)]
        #     self.num_episodes = 0
        #     self.dist_paths = [list() for i in range(self.num_actors)]

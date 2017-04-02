import itertools
import numpy as np
from collections import defaultdict


def array_to_string(state):
    return ''.join((chr(s + 128) for s in state))


def string_to_array(state_string):
    return np.array((ord(s) - 128 for s in state_string))


def e_greedy_decision(best_action, epsilon, nA):
    A = np.ones(nA, dtype=float) * epsilon / (nA)
    A[best_action] += (1.0 - epsilon)
    return A


class EGreedyPolicyModel(object):
    def __init__(self, nA, eps_start=1.0, eps_end=0.01, eps_length=1000):
        self.i_episode = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_length = eps_length
        self.eps_decay = (eps_start - eps_end) / eps_length
        self.curr_eps = eps_start
        self.nA = nA

    def sample_action(self, best_action):
        action_probs = e_greedy_decision(best_action=best_action, epsilon=self.curr_eps, nA=self.nA)
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)

    def update_eps(self):
        self.i_episode += 1
        self.curr_eps = max(self.eps_start - self.eps_decay * self.i_episode, self.eps_end)


class TabQModel(object):
    def __init__(self, n_action, gamma=1.0, alpha=0.5):
        self.Q = defaultdict(lambda: np.zeros(n_action))
        self.gamma = gamma
        self.alpha = alpha

    def value_at(self, state, action):
        return self.Q[state][action]

    def best_action(self, state):
        return np.argmax(self.Q[state])

    def td_update(self, state, action, reward_cumulated, N, done, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        if done:
            td_target = reward_cumulated
        else:
            td_target = reward_cumulated + (self.gamma ** N) * self.Q[next_state][best_next_action]
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta


class Option(object):
    def __init__(self, state_rep, actions, end_func, r_in,
                 gamma=1.0, alpha=0.5,
                 eps_start=1.0, eps_end=0.01, eps_length=1000, is_learning=True):
        self.caller = None
        self.actions = actions
        self.nA = len(actions)
        self.should_end = end_func
        self.intrinsic_reward = r_in
        self.get_abstract_state = state_rep
        self.q_model = TabQModel(n_action=self.nA, gamma=gamma, alpha=alpha)
        self.gamma = gamma
        self.policy = EGreedyPolicyModel(nA=self.nA,
                                         eps_start=eps_start,
                                         eps_end=eps_end,
                                         eps_length=eps_length)
        self.curr_state = None
        self.curr_action = None
        self.reward_cumulated = 0
        self.episode_count = 0
        self.debug_output = False
        self.is_learning = is_learning

    def make_decision(self, state):
        """

        Parameters
        ----------
        state
        call_stack: list

        Returns
        -------

        """
        self.curr_state = state
        state_string = array_to_string(state)
        best_action = self.q_model.best_action(state=state_string)
        self.curr_action = self.policy.sample_action(best_action)
        if self.debug_output:
            print("action:\t{}-------------------------------------------------------------".format(self.curr_action))
        return self.curr_action

    def act_recursively(self, raw_state, working_option):
        state = self.get_abstract_state(raw_state)
        sampled_action = self.actions[self.make_decision(state)]
        if not isinstance(sampled_action, Option):
            return sampled_action
        sampled_action.caller = self
        working_option[0] = sampled_action
        return sampled_action.act_recursively(raw_state, working_option)

    def learn(self, raw_next_state, reward, done, N):
        if not self.is_learning:
            return
        next_state = self.get_abstract_state(raw_next_state)
        state_string = array_to_string(self.curr_state)
        next_state_string = array_to_string(next_state)
        self.q_model.td_update(state=state_string, action=self.curr_action, reward_cumulated=reward,
                               N=N, done=done, next_state=next_state_string)
        if done:
            self.policy.update_eps()

    def learn_option(self, raw_next_state, real_reward, real_done, working_option):
        # assert isinstance(self.actions[action], int)
        is_child = (self.caller is not None)
        if is_child:
            self.reward_cumulated += self.gamma ** self.episode_count * real_reward
            self.episode_count += 1
            h_done = self.should_end(self.curr_state, self.curr_action) or real_done
            if h_done:
                reward = self.intrinsic_reward(self.curr_state, self.curr_action)
            else:
                reward = real_reward
        else:
            reward = real_reward
            h_done = real_done
        self.learn(raw_next_state, reward, h_done, 1)
        if is_child and h_done:
            self.caller.learn(raw_next_state, self.reward_cumulated, real_done, self.episode_count)
            self.reward_cumulated = 0
            self.episode_count = 0
            working_option[0] = self.caller


class HierarchicalAgent(object):
    def __init__(self, root_option):
        self.root = root_option
        self.working_option = [root_option]

    def make_decision(self, state):
        real_action = self.working_option[0].act_recursively(state, self.working_option)
        return real_action

    def receive_feedback(self, next_state, reward, done):
        self.working_option[0].learn_option(next_state, reward, done, self.working_option)

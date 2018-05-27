import gym
import multiprocessing as mp
import queue
import logging
import numpy as np
from arena.experiment import Experiment

class Agent(object):
    def __init__(self, observation_space, action_space,
                 shared_params, stats_rx, acts_tx,
                 is_learning, global_t, pid=0, **kwargs):
        """

        Parameters
        ----------
        observation_space : gym.Space
        action_space : gym.Space
        """
        self.observation_space = observation_space
        if isinstance(observation_space, list):
            self.multi_obs = True
            self.copy_obs = self.copy_multi
        else:
            self.copy_obs = self.copy_single
            self.multi_obs = False
        self.action_space = action_space
        self.params = shared_params
        self.stats_rx = stats_rx
        self.acts_tx = acts_tx
        self.is_learning = is_learning
        self.terminated = False
        self.current_obs = None
        self.current_action = None
        self.current_reward = None
        self.current_episode_ends = None
        self.current_info = None
        self.gb_t = global_t
        self.id = pid
        # logging.debug("Agent {} initialized".format(self.id))

    def clean(self):
        pass

    def copy_multi(self, rxmsg):
        self.current_obs = list(map(np.copy, rxmsg))

    def copy_single(self, rxmsg):
        self.current_obs = rxmsg.copy()

    def reset(self):
        self.current_obs = None
        self.current_action = None
        self.current_reward = None
        self.current_episode_ends = None
        self.current_info = None

    def terminate(self):
        self.terminated = True
        # logging.debug("Agent: {} terminated".format(self.id))

    def run_loop(self):
        while not self.terminated:
            # logging.debug("Agent: {} waiting for observation".format(self.id))

            rx_msg_state = self.stats_rx[0].recv(timeout=1.0)
            while rx_msg_state is None:
                if Experiment.is_terminated:
                    self.terminate()
                    return -1
                rx_msg_state = self.stats_rx[0].recv(timeout=5.0)

            try:
                # self.current_obs = rx_msg["observation"].copy()
                self.copy_obs(rx_msg_state["observation"])
            except KeyError:
                raise ValueError("Failed to receive observation")

            self.current_action = self.act(self.current_obs)
            self.acts_tx.send({"action": self.current_action})
            rx_msg_feedback = self.stats_rx[1].recv(timeout=1.0)
            while rx_msg_feedback is None:
                if Experiment.is_terminated:
                    self.terminate()
                    return -1
                rx_msg_feedback = self.stats_rx[1].recv(timeout=5.0)
            try:
                self.current_reward = np.asscalar(rx_msg_feedback["reward"])
                self.current_episode_ends = np.asscalar(rx_msg_feedback["done"])
                info_dict = rx_msg_feedback.copy()
                info_dict.pop("reward", None)
                info_dict.pop("done", None)
                self.current_info = {k: v.copy() for (k, v) in list(info_dict.items())}
                # logging.debug("rk {} ".format(rx_msg))
            except KeyError:
                raise ValueError("Failed to receive feedback in self.stats_rx")
            self.receive_feedback(self.current_reward, self.current_episode_ends, info=self.current_info)

    def act(self, observation):
        """

        Parameters
        ----------
        observation : gym.Space
        Returns
        -------

        """
        return self.action_space.sample()

    def receive_feedback(self, reward, done, info={}):
        pass

    def stats_keys(self):
        return []

    def stats_values(self):
        return []






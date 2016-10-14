import gym
import multiprocessing as mp
import queue
from arena.utils import  ProcessState
import logging


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
        self.obs_space = observation_space
        self.action_space = action_space
        self.params = shared_params
        self.stats_rx = stats_rx
        self.acts_tx = acts_tx
        self.is_learning = is_learning
        self.terminated = False
        self.current_obs = None
        self.current_action = None
        self.reward = None
        self.episode_ends = None
        self.gb_t = global_t
        self.lc_t = 0
        self.id = pid
        logging.debug("Agent {} initialized".format(self.id))

    def reset(self):
        self.current_obs = None
        self.current_action = None
        self.reward = None
        self.episode_ends = None

    def run_loop(self):
        while not self.terminated:
            # logging.debug("Agent: {} waiting for observation".format(self.id))
            rx_msg = self.stats_rx.recv()
            try:
                self.current_obs = rx_msg["observation"]
            except KeyError:
                raise ValueError("Failed to receive observation")

            self.current_action = self.act(self.current_obs)
            self.acts_tx.send(self.current_action)
            rx_msg = self.stats_rx.recv()
            try:
                self.reward = rx_msg["reward"]
                self.episode_ends = rx_msg["done"]
            except KeyError:
                raise ValueError("Failed to receive feedback in self.stats_rx")
            self.receive_feedback(self.reward, self.episode_ends)
            self.lc_t += 1

    def act(self, observation):
        """

        Parameters
        ----------
        observation : gym.Space
        Returns
        -------

        """
        raise NotImplementedError

    def receive_feedback(self, reward, done):
        raise NotImplementedError

    def stats_keys(self):
        return []

    def stats_values(self):
        return []


class RandomAgent(Agent):
    def __init__(self,  observation_space, action_space,
                 shared_params, stats_rx: mp.Queue,acts_tx: mp.Queue,
                 is_learning, global_t, pid=0, **kwargs):
        """

        Parameters
        ----------
        observation_space : gym.Space
        action_space : gym.Space
        """
        super(RandomAgent, self).__init__(
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid, **kwargs
        )

    def act(self, observation):
        """

        Parameters
        ----------
        observation : gym.Space

        Returns
        -------
        """

        return self.action_space.sample()

    def receive_feedback(self, reward, done):
        pass


import gym
import numpy as np
import multiprocessing as mp
import queue
from arena.utils import ProcessState

class Actuator(object):
    def __init__(self, func_get_env, stats_tx: mp.Queue, acts_rx: mp.Queue,
                 cmd_signal: mp.Queue, episode_data_q: mp.Queue,
                 global_t, act_id=0):
        self.env = func_get_env()
        self.stats_tx = stats_tx
        self.acts_rx = acts_rx
        self.signal = cmd_signal
        self.is_idle = True
        self.is_terminated = False
        self.current_obs = None
        self.action = None
        self.reward = None
        self.episode_ends = None
        self.reset()
        self.episode_q = episode_data_q
        self.episode_count = 0
        self.episode_reward = 0
        self.id = act_id
        self.gb_t = global_t

    def reset(self):
        self.current_obs = self.env.reset()
        self.action = None
        self.reward = None
        self.episode_ends = None
        self.episode_count = 0
        self.episode_reward = 0

    def receive_cmd(self):
        if self.is_idle:
            cmd = self.signal.get(block=True)
        else:
            try:
                cmd = self.signal.get(block=False)
            except queue.Empty:
                cmd = None
        if cmd is not None:
            if cmd == ProcessState.terminate:
                self.is_terminated = True
            elif cmd == ProcessState.stop:
                self.is_idle = True
            elif cmd == ProcessState.start:
                self.is_idle = False
                self.reset()
            else:
                raise ValueError("Unknown command from self.signal")

    def run_loop(self):
        while not self.is_terminated:
            self.receive_cmd()
            self.stats_tx.put({"observation": self.current_obs}, block=True)
            current_action = self.acts_rx.get(block=True)
            self.current_obs, self.reward, self.episode_ends, info_env = \
                self.env.step(current_action)
            self.stats_tx.put({"reward": self.reward, "done": self.episode_ends},
                              block=True)
            self.episode_reward += self.reward
            self.episode_count += 1
            self.gb_t += 1

            if self.episode_ends:
                self.episode_q.put(
                    {"id": self.id, "episode_reward": self.episode_reward,
                     "episode_count": self.episode_count},
                    block=True
                )
                self.reset()







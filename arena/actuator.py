import gym
import numpy as np
import multiprocessing as mp
import queue
from arena.mp_utils import ProcessState
import logging


class Actuator(object):
    def __init__(self, func_get_env, stats_tx, acts_rx,
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

        logging.debug("Actuator: {} initialized".format(self.id))

    def reset(self):
        self.current_obs = self.env.reset()
        self.action = None
        self.reward = None
        self.episode_ends = None
        self.episode_count = 0
        self.episode_reward = 0

    def receive_cmd(self):
        if self.is_idle:
            logging.debug("Actuator: {} waiting for start".format(self.id))
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
                self.episode_q.put(
                    {"id": self.id, "status": ProcessState.stop}
                )
                self.receive_cmd()
            elif cmd == ProcessState.start:
                self.is_idle = False
                self.reset()
                logging.debug("Actuator: {} started".format(self.id))
            else:
                raise ValueError("Unknown command from self.signal")

    def run_loop(self):
        while not self.is_terminated:
            self.receive_cmd()
            if self.is_terminated:
                break
            self.stats_tx[0].send({"observation": self.current_obs})
            # if not self.acts_rx.poll(timeout=10 * 60):
            #     logging.warning("Not received action for too long, potential error")
            #     break
            received_dict = self.acts_rx.recv()
            current_action = received_dict["action"]
            self.current_obs, self.reward, self.episode_ends, info_env = \
                self.env.step(current_action)
            self.stats_tx[1].send({"reward": self.reward, "done": self.episode_ends})
            self.episode_reward += self.reward
            self.episode_count += 1

            if self.episode_ends:
                self.episode_q.put(
                    {"id": self.id, "episode_reward": self.episode_reward,
                     "episode_count": self.episode_count},
                    block=True
                )
                self.reset()







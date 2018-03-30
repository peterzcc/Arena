from arena.memory import AcMemory
from arena.agents import Agent
# from trpo_model import TrpoModel
from policy_gradient_model import PolicyGradientModel
import numpy as np
import logging
from dict_memory import DictMemory
import threading
import time
import gc
from ram_util import resident


class AsyncAgent(Agent):
    def __init__(self, observation_space, action_space,
                 shared_params, stats_rx, acts_tx,
                 is_learning, global_t, pid=0,
                 ):
        Agent.__init__(
            self,
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid
        )

        assert shared_params is not None
        # self.param_lock = shared_params["lock"]
        self.policy: PolicyGradientModel = shared_params["models"][0]
        self.memory: DictMemory = shared_params["memory"]
        self.train_lock = threading.Lock()
        assert not self.policy.parallel_predict
        self.num_epoch = 0
        self.global_t = 0
        self.batch_start_time = None
        self.num_episodes = 0
        self.train_data = None

    def act(self, observation):

        if self.batch_start_time is None:
            self.batch_start_time = time.time()
        processed_observation = [observation[0]]
        if len(observation) == 2:
            processed_observation.append(observation[1])
        action, agent_info = self.policy.predict(processed_observation, pid=self.id)

        self.memory.append_state(observation, action, info=agent_info, pid=self.id)

        return action

    def receive_feedback(self, reward, done, info={}):

        self.memory.append_feedback(reward, pid=self.id)
        # is_episode_clipped = self.time_count * self.model.num_actors == self.model.batch_size
        batch_ends = self.memory.incre_count_and_check_done(self.id)
        if done or batch_ends:
            terminated = False
            if done:
                try:
                    terminated = np.asscalar(info["terminated"])
                except KeyError:
                    logging.debug("warning: no info about real termination ")
                    terminated = done
            self.memory.transfer_single_path(terminated, self.id)
        if batch_ends:
            with self.train_lock:
                extracted_result = self.memory.profile_extract_all(self.id)
                train_before = time.time()
                self.policy.train(extracted_result["paths"], pid=self.id)
                train_after = time.time()
                train_time = (train_after - train_before) / extracted_result["time_count"]
                fps = 1.0 / train_time
                res_ram = resident() / (1024 * 1024)
                if self.id == 0:
                    logging.info(
                        '\nfps:%f\ntt:%f\nram:%f\n\n\n\n' \
                        % (
                            fps,
                            train_time,
                            res_ram
                        ))

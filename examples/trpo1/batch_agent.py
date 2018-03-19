from arena.memory import AcMemory
from arena.agents import Agent
# from trpo_model import TrpoModel
from policy_gradient_model import PolicyGradientModel
import numpy as np
import logging
from dict_memory import DictMemory
import time
import gc
from ram_util import resident


class BatchUpdateAgent(Agent):
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
        self.model: PolicyGradientModel = shared_params["models"][0]
        self.memory: DictMemory = shared_params["memory"]

        self.num_epoch = 0
        self.global_t = 0
        self.batch_start_time = None
        # max_l = 10000

        self.time_count = 0

        self.num_episodes = 0
        self.train_data = None
        self.should_clip_episodes = (self.model.batch_mode == "timestep")

    def act(self, observation):

        if self.batch_start_time is None:
            self.batch_start_time = time.time()
        processed_observation = [observation[0]]
        if len(observation) == 2:
            processed_observation.append(observation[1])
        action, agent_info = self.model.predict(processed_observation, pid=self.id)

        self.memory.append_state(observation, action, info=agent_info, pid=self.id)

        return action

    def receive_feedback(self, reward, done, info={}):

        self.memory.append_feedback(reward, pid=self.id)
        self.time_count += 1
        is_episode_clipped = self.time_count * self.model.num_actors == self.model.batch_size
        if done or (self.should_clip_episodes and is_episode_clipped):

            terminated = False
            if done:
                try:
                    terminated = np.asscalar(info["terminated"])
                except KeyError:
                    logging.debug("warning: no info about real termination ")
                    terminated = done
            extracted_result = self.memory.add_path(terminated, pid=self.id)
            if extracted_result is not None:
                train_before = time.time()
                self.model.train(extracted_result["paths"])
                train_after = time.time()
                train_time = (train_after - train_before) / extracted_result["time_count"]
                fps = 1.0 / train_time
                res_ram = resident() / (1024 * 1024)
                logging.info(
                    '\nfps:%f\ntt:%f\nram:%f\n\n\n\n' \
                    % (
                        fps,
                        train_time,
                        res_ram
                       ))
            if is_episode_clipped:
                self.model.batch_barrier.wait()
                self.time_count = 0




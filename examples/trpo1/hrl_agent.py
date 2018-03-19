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


class HrlAgent(Agent):
    def __init__(self, observation_space, action_space,
                 shared_params, stats_rx, acts_tx,
                 is_learning, global_t, pid=0,
                 full_tasks=[],
                 ):
        Agent.__init__(
            self,
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid
        )

        assert shared_params is not None
        # self.param_lock = shared_params["lock"]
        self.root_policy: PolicyGradientModel = shared_params["models"][0]
        self.sub_policies = shared_params["models"][1:]
        self.sub_pol_act_t = 0
        self.acting_policy = self.root_policy
        self.memory: DictMemory = shared_params["memory"]
        self.f_should_return_root = shared_params["f_should_return_root"]

        self.num_epoch = 0
        self.global_t = 0
        self.batch_start_time = None
        # max_l = 10000

        self.time_count = 0

        self.num_episodes = 0
        self.train_data = None
        self.should_clip_episodes = (self.root_policy.batch_mode == "timestep")
        self.full_tasks = full_tasks

    def act(self, observation):
        # TODO: hrl

        if self.batch_start_time is None:
            self.batch_start_time = time.time()

        if self.acting_policy == self.root_policy:
            root_action, root_model_info = self.root_policy.predict(observation, pid=self.id)
            self.memory.append_observation(observation, pid=self.id)
            self.memory.append_action(root_action, root_model_info, pid=self.id)
            self.acting_policy = self.sub_policies[root_action]
        action, model_info = self.acting_policy.predict(observation, pid=self.id)
        self.sub_pol_act_t += 1

        return action

    def receive_feedback(self, reward, done, info={}):
        # TODO: HRL

        self.memory.append_feedback(reward, pid=self.id)

        should_return_root = self.acting_policy != self.root_policy and \
                             self.f_should_return_root(self.acting_policy, self.sub_pol_act_t, self.root_policy)
        batch_ends = False

        # reset the agent in control if necessary
        if should_return_root:
            self.acting_policy = self.root_policy
            self.sub_pol_act_t = 0
            self.time_count += 1

        batch_ends = self.memory.incre_count_and_check_done()

        if done or batch_ends:
            self.acting_policy = self.root_policy
            self.sub_pol_act_t = 0

            try:
                terminated = np.asscalar(info["terminated"])
            except KeyError:
                logging.debug("warning: no info about real termination ")
                terminated = done
            finally:
                terminated = False
            self.memory.transfer_single_path(terminated, self.id)
        if batch_ends:
            self.root_policy.batch_barrier.wait()
            if self.id == 0:
                extracted_result = self.memory.profile_extract_all()
                train_before = time.time()
                self.root_policy.train(extracted_result["paths"])
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
            self.root_policy.batch_barrier.wait()
            self.time_count = 0

from arena.agents import Agent
# from trpo_model import TrpoModel
from policy_gradient_model import PolicyGradientModel
import numpy as np
import logging
from dict_memory import DictMemory
import time
import gc
from ram_util import resident


class FlexibleHrlAgent(Agent):
    def __init__(self, observation_space, action_space,
                 shared_params, stats_rx, acts_tx,
                 is_learning, global_t, pid=0,
                 full_tasks=[],
                 should_train_subpolicy=lambda i: False,
                 ):
        Agent.__init__(
            self,
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid
        )

        assert shared_params is not None
        # self.param_lock = shared_params["lock"]
        self.decider: PolicyGradientModel = shared_params["models"]["decider"]
        self.switcher: PolicyGradientModel = shared_params["models"]["switcher"]
        self.sub_policies = shared_params["models"]["leafs"]
        self.sub_pol_act_t = 0
        self.memory: DictMemory = shared_params["memory"]

        self.num_epoch = 0
        self.global_t = 0
        self.batch_start_time = None
        # max_l = 10000

        self.time_count = 0

        self.num_episodes = 0
        self.train_data = None
        self.should_clip_episodes = (self.decider.batch_mode == "timestep")
        self.full_tasks = full_tasks
        self.current_policy_id = None
        self.is_initial_step = 1

    def update_meta_status(self, should_switch):
        self.current_policy_id = None if should_switch else self.current_policy_id
        self.sub_pol_act_t = 1 if should_switch else self.sub_pol_act_t + 1

    def wrap_meta_obs(self, observation):
        return [*observation, np.array([self.current_policy_id, self.sub_pol_act_t, self.is_initial_step])]

    def act(self, observation):

        if self.batch_start_time is None:
            self.batch_start_time = time.time()

        self.is_initial_step = False
        wrapped_obs = self.wrap_meta_obs(observation)

        if self.current_policy_id is None:
            decision, decider_model_info = self.decider.predict(wrapped_obs, pid=self.id)
            self.current_policy_id = decision
            self.sub_pol_act_t = 1
        else:
            decision, decider_model_info = None, None
        should_switch, switcher_model_info = self.switcher.predict(wrapped_obs, pid=self.id)

        action, leaf_model_info = self.sub_policies[self.current_policy_id].predict(observation, pid=self.id)
        self.memory.append_hrl_state(wrapped_obs, should_switch, switcher_model_info, decision, decider_model_info,
                                     self.id,
                                     leaf_id=self.current_policy_id, leaf_action=action,
                                     leaf_model_info=leaf_model_info,
                                     curr_time_step=self.time_count)
        self.update_meta_status(should_switch)

        return action

    def sync_train(self):
        self.decider.batch_barrier.wait()
        if self.id == 0:
            extracted_result = self.memory.profile_extract_all(with_subtasks=True)
            train_before = time.time()
            self.decider.train(extracted_result["decider_data"])
            self.switcher.train(extracted_result["switcher_data"])

            for p, train_paths in zip(self.sub_policies, extracted_result["leaf_data"]):
                p.train(train_paths)

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
        self.decider.batch_barrier.wait()
        self.num_epoch += 1

    def receive_feedback(self, reward, done, info={}):

        self.memory.append_feedback(reward, pid=self.id, info=info)

        self.time_count += 1
        if done:
            self.time_count = 0
            self.sub_pol_act_t = 0
            self.current_policy_id = None

        if done or self.current_policy_id is None:
            batch_ends = self.memory.check_done()
        else:
            batch_ends = False

        if done or batch_ends:
            terminated = False
            try:
                terminated = np.asscalar(info["terminated"])
            except KeyError:
                logging.warning(": no info about real termination ")
                terminated = done
            self.memory.transfer_single_path(terminated, self.id)
        self.decider.batch_ends[self.id] = batch_ends
        if batch_ends:
            self.sync_train()

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
        self.root_policy: PolicyGradientModel = shared_params["models"][0]
        self.sub_policies = shared_params["models"][1:]
        self.sub_pol_act_t = 0
        self.memory: DictMemory = shared_params["memory"]

        self.num_epoch = 0
        self.global_t = 0
        self.batch_start_time = None
        # max_l = 10000

        self.time_count = 0

        self.num_episodes = 0
        self.train_data = None
        self.should_clip_episodes = (self.root_policy.batch_mode == "timestep")
        self.full_tasks = full_tasks
        self.current_policy_id = 0
        self.is_initial_step = 1

    def update_meta_status(self, root_decision):
        self.current_policy_id = root_decision - 1 if root_decision != 0 else self.current_policy_id
        self.sub_pol_act_t = 1 if root_decision != 0 else self.sub_pol_act_t + 1

    def wrap_meta_obs(self, observation):
        return [*observation, np.array([self.current_policy_id, self.sub_pol_act_t, self.is_initial_step])]

    def act(self, observation):

        if self.batch_start_time is None:
            self.batch_start_time = time.time()

        self.is_initial_step = (self.time_count == 0)
        wrapped_obs = self.wrap_meta_obs(observation)

        root_decision, root_model_info = self.root_policy.predict(wrapped_obs, pid=self.id)
        # self.memory.append_observation(wrapped_obs, pid=self.id)
        # self.memory.append_action(root_decision, root_model_info, pid=self.id)

        action, leaf_model_info = self.sub_policies[self.current_policy_id].predict(observation, pid=self.id)
        self.update_meta_status(root_decision)
        self.memory.append_state(wrapped_obs, root_decision, root_model_info, self.id,
                                 leaf_id=self.current_policy_id, leaf_action=action, leaf_model_info=leaf_model_info,
                                 curr_time_step=self.time_count)



        return action

    def receive_feedback(self, reward, done, info={}):

        self.memory.append_feedback(reward, pid=self.id, info=info)

        self.time_count += 1
        if done:
            self.time_count = 0
            self.sub_pol_act_t = 1

        batch_ends = self.memory.incre_count_and_check_done()

        if done or batch_ends:
            terminated = False
            try:
                terminated = np.asscalar(info["terminated"])
            except KeyError:
                logging.warning(": no info about real termination ")
                terminated = done
            self.memory.transfer_single_path(terminated, self.id)
        self.root_policy.batch_ends[self.id] = batch_ends
        if batch_ends:
            self.root_policy.batch_barrier.wait()
            if self.id == 0:
                extracted_result = self.memory.profile_extract_all(with_subtasks=True)
                train_before = time.time()
                self.root_policy.train(extracted_result["paths"])

                for p, train_paths in zip(self.sub_policies, extracted_result["leaf_paths"]):
                    if train_paths is not None:
                        p.train(train_paths)
                    else:
                        logging.debug("No training data for {}".format(p.name))

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
            self.num_epoch += 1

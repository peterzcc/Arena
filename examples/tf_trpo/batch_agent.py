from arena.memory import AcMemory
from arena.agents import Agent
from trpo_model import TrpoModel
import numpy as np
import logging


class BatchUpdateAgent(Agent):
    def __init__(self, observation_space, action_space,
                 shared_params, stats_rx, acts_tx,
                 is_learning, global_t, pid=0,
                 model=None,
                 batch_size=1,
                 discount=0.995,
                 lam=0.97
                 ):
        Agent.__init__(
            self,
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid
        )

        self.action_dimension = action_space.low.shape[0]
        self.state_dimension = observation_space.low.shape[0]

        if shared_params is None:
            pass
        else:
            self.param_lock = shared_params["lock"]
            self.global_model = shared_params["global_model"]
        self.model = model

        # Constant vars
        self.batch_size = batch_size
        self.discount = discount

        # State information
        self.counter = batch_size
        self.episode_step = 0
        self.epoch_reward = 0
        max_l = 10000
        # if model is None: TODO: recover
        self.model = TrpoModel(self.observation_space, self.action_space)
        # else:
        #     self.model = model
        self.memory = AcMemory(observation_shape=self.observation_space.shape,
                               action_shape=self.action_space.shape,
                               max_size=batch_size + max_l,
                               gamma=self.discount,
                               lam=lam,
                               use_gae=True,
                               get_critic_online=False,
                               info_shape=self.model.info_shape)

        self.num_episodes = 0

    def act(self, observation):
        # logging.debug("rx obs: {}".format(observation))
        # if self.memory.Tmax == 0:
        #     #TODO: sync from global model
        #     pass

        # TODO: Implement this predict
        action, agent_info = self.model.predict(observation)
        final_action = \
            np.clip(action, self.action_space.low, self.action_space.high).flatten()

        self.memory.append_state(observation, action, info=agent_info)

        # logging.debug("tx a: {}".format(action))
        return final_action

    def receive_feedback(self, reward, done):
        # logging.debug("rx r: {} \td:{}".format(reward, done))

        self.memory.append_feedback(reward)
        self.episode_step += 1
        self.epoch_reward += reward
        if done:
            self.counter -= self.episode_step
            self.memory.fill_episode_critic(self.model.compute_critic)
            self.memory.add_path(done)
            self.num_episodes += 1
            self.episode_step = 0
            if self.counter <= 0:
                self.train_once()

                logging.info(
                    'Thd[%d] Average Return:%f,  Num Traj:%d ' \
                    % (self.id,
                       self.epoch_reward / self.num_episodes,
                       self.num_episodes,
                       ))

                self.memory.reset()
                self.epoch_reward = 0
                self.counter = self.batch_size
                self.num_episodes = 0

    def train_once(self):
        train_data = self.memory.extract_all()
        diff, new = self.model.compute_update(train_data)
        self.model.update(diff=diff, new=new)
        # with self.param_lock: TODO: async
        #     self.global_model.update(diff)

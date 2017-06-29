from arena.memory import AcMemory
from arena.agents import Agent
# from trpo_model import TrpoModel
from multi_trpo_model import MultiTrpoModel
import numpy as np
import logging
from dict_memory import DictMemory
import time


class BatchUpdateAgent(Agent):
    def __init__(self, observation_space, action_space,
                 shared_params, stats_rx, acts_tx,
                 is_learning, global_t, pid=0,
                 model=None,
                 batch_size=1,
                 discount=0.995,
                 lam=0.97,
                 timestep_limit=1000
                 ):
        Agent.__init__(
            self,
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid
        )
        # self.use_filter = False
        # if self.use_filter:
        #     self.obsfilter = ZFilter(observation_space.shape, clip=5)
        #     self.rewfilter = ZFilter((), demean=False, clip=10)
        # else:
        #     self.obsfilter = IDENTITY
        #     self.rewfilter = IDENTITY

        # self.action_dimension = action_space.low.shape[0]
        # self.state_dimension = observation_space.low.shape[0]

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
        self.num_epoch = 0
        self.counter = batch_size
        self.episode_step = 0
        self.epoch_reward = 0
        # max_l = 10000

        if model is None:
            self.model = MultiTrpoModel(self.observation_space, self.action_space,
                                        timestep_limit=timestep_limit)
        else:
            self.model = model
        # self.memory = AcMemory(observation_shape=self.observation_space.shape,
        #                        action_shape=self.action_space.shape,
        #                        max_size=batch_size + max_l,
        #                        gamma=self.discount,
        #                        lam=lam,
        #                        use_gae=False,
        #                        get_critic_online=False,
        #                        info_shape=self.model.info_shape)
        self.memory = DictMemory(gamma=self.discount, lam=lam, use_gae=True, normalize=True,
                                 timestep_limit=timestep_limit)

        self.num_episodes = 0

    def act(self, observation):
        # logging.debug("rx obs: {}".format(observation))
        # if self.memory.Tmax == 0:
        #     #TODO: sync from global model
        #     pass

        # TODO: Implement this predict
        # if self.image_input:
        #     observation = observation.astype(np.float32) / 255.0
        # observation = self.obsfilter(observation)
        action, agent_info = self.model.predict(observation)
        # final_action = \
        #     np.clip(action, self.action_space.low, self.action_space.high).flatten()

        self.memory.append_state(observation, action, info=agent_info)

        # logging.debug("tx a: {}".format(action))
        # print("action: "+str(final_action))
        return action

    def receive_feedback(self, reward, done, info={}):
        # logging.debug("rx r: {} \td:{}".format(reward, done))
        self.epoch_reward += reward
        # reward = self.rewfilter(reward)

        self.memory.append_feedback(reward)
        self.episode_step += 1

        if done:
            self.counter -= self.episode_step
            self.memory.fill_episode_critic(self.model.compute_critic)
            try:
                terminated = np.asscalar(info["terminated"])
                # logging.debug("terminated {} ".format(terminated))
            except KeyError:
                logging.debug("warning: no info about real termination ")
                terminated = done
            self.memory.add_path(terminated)
            self.num_episodes += 1
            self.episode_step = 0
            if self.counter <= 0:
                train_before = time.time()
                self.train_once()
                train_after = time.time()
                fps = (self.batch_size - self.counter) / (train_after - train_before)

                logging.info(
                    'Epoch:%d \nThd[%d] \nAverage Return:%f,  \nNum Traj:%d \nfps:%f \n' \
                    % (self.num_epoch,
                       self.id,
                       self.epoch_reward / self.num_episodes,
                       self.num_episodes,
                       fps
                       ))

                self.memory.reset()
                self.epoch_reward = 0
                self.counter = self.batch_size
                self.num_episodes = 0
                self.num_epoch += 1

    def train_once(self):
        train_data = self.memory.extract_all()
        diff, new = self.model.compute_update(train_data)
        self.model.update(diff=diff, new=new) \
            # with self.param_lock: TODO: async
        #     self.global_model.update(diff)


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


def IDENTITY(x):
    return x

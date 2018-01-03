from arena.memory import AcMemory
from arena.agents import Agent
# from trpo_model import TrpoModel
from multi_trpo_model import MultiTrpoModel
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
                 model=None,
                 timestep_limit=1000,
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
        self.model = model
        if shared_params is None:
            pass
        else:
            # self.param_lock = shared_params["lock"]
            self.global_model = shared_params["global_model"]
            self.model = self.global_model

        # Constant vars

        # self.discount = discount

        # State information
        self.num_epoch = 0
        self.global_t = 0
        self.batch_start_time = None
        # max_l = 10000

        # if model is None:
        #     self.model = MultiTrpoModel(self.observation_space, self.action_space,
        #                                 timestep_limit=timestep_limit,
        #                                 cg_damping=0.1,
        #                                 max_kl=0.01,
        #                                 cg_iters=10)
        # else:
        #     self.model = model
        # self.memory = AcMemory(observation_shape=self.observation_space.shape,
        #                        action_shape=self.action_space.shape,
        #                        max_size=batch_size + max_l,
        #                        gamma=self.discount,
        #                        lam=lam,
        #                        use_gae=False,
        #                        get_critic_online=False,
        #                        info_shape=self.model.info_shape)
        self.time_count = 0
        # if self.model.batch_mode == 'timestep':
        #     self.thread_batch_size = self.model.batch_size / self.model.num_actors
        #     assert self.model.batch_size % self.model.num_actors == 0
        # else:
        #     self.thread_batch_size = np.inf
        self.num_episodes = 0
        self.train_data = None
        self.should_clip_episodes = (self.model.batch_mode == "timestep")

    def act(self, observation):
        # logging.debug("rx obs: {}".format(observation))
        # if self.memory.Tmax == 0:
        #     #TODO: sync from global model
        #     pass
        if self.batch_start_time is None:
            self.batch_start_time = time.time()
        processed_observation = [observation[0]]
        if len(observation) == 2:
            processed_observation.append(observation[1].astype(np.float32) / 255.0)
        action, agent_info = self.model.predict(processed_observation, pid=self.id)
        # action = self.action_space.sample()
        # agent_info = {}
        # final_action = \
        #     np.clip(action, self.action_space.low, self.action_space.high).flatten()

        self.model.memory.append_state(observation, action, info=agent_info, pid=self.id)

        # logging.debug("tx a: {}".format(action))
        # print("action: "+str(final_action))
        return action

    def receive_feedback(self, reward, done, info={}):
        # logging.debug("rx r: {} \td:{}".format(reward, done))
        # reward = self.rewfilter(reward)

        self.model.memory.append_feedback(reward, pid=self.id)
        self.time_count += 1
        # assert self.time_count <= self.thread_batch_size
        is_episode_clipped = self.time_count * self.model.num_actors == self.model.batch_size
        if done or (self.should_clip_episodes and is_episode_clipped):
            # self.counter -= self.episode_step
            # self.global_t += self.episode_step
            terminated = False
            if done:
                try:
                    terminated = np.asscalar(info["terminated"])
                    # logging.debug("terminated {} ".format(terminated))
                except KeyError:
                    logging.debug("warning: no info about real termination ")
                    terminated = done
            extracted_result = self.model.memory.add_path(terminated, pid=self.id)
            if extracted_result is not None:
                train_before = time.time()
                self.model.compute_update(extracted_result["paths"])
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


    def train_once(self):

        diff, new = self.model.compute_update(self.train_data)
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

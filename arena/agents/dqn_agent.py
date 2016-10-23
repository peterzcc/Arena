from arena.agents import Agent
from arena.replay_memory import create_memory,ReplayMemory
from arena.base import Base
from arena.e_greedy import EpsGreedy
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import gym
import gym.spaces
import cv2
import logging

class DqnAgent(Agent):
    def __init__(self, observation_space, action_space,
                 shared_params, stats_rx, acts_tx,
                 is_learning, global_t, pid=0,
                 f_get_sym = None,
                 is_double_q=False,
                 replay_memory_size=1000000,
                 train_start=100,
                 history_length=4,
                 training_interval=4,
                 minibatch_size=32,
                 optimizer=mx.optimizer.create(name='adagrad', learning_rate=0.01, eps=0.01),
                 policy=None,
                 initializer=None,
                 discount=0.99,
                 freeze_interval=10000,
                 ctx=mx.cpu()
                 ):
        if not isinstance(action_space,gym.spaces.Discrete):
            raise ValueError("Dqn only supports single dimensional discrete action space")
        super(DqnAgent, self).__init__(
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid
        )
        self.minibatch_size = minibatch_size
        self.data_shapes = {"data": (minibatch_size, history_length)+observation_space.shape,
                            "dqn_action": (minibatch_size,), "dqn_reward": (minibatch_size,)}
        dqn_sym = f_get_sym(action_space.n)
        self.is_double_q = is_double_q
        self.ctx = ctx
        if self.params is None:
            shared_qnet_params=None
        elif "qnet" in self.params:
            shared_qnet_params = self.params["qnet"]
        self.qnet = Base(data_shapes=self.data_shapes, sym_gen=dqn_sym, name="QNet",
                    initializer=initializer, params=shared_qnet_params, ctx=ctx)
        self.target_qnet = self.qnet.copy(name="TargetQNet", ctx=ctx)
        if policy is None:
            policy = EpsGreedy(1, 0.1, 4000000, 1)
        self.policy = policy
        self.training_interval = training_interval
        self.local_steps = 0
        self.train_start = train_start
        self.force_explore = False
        self.discount = discount

        self.memory = create_memory(observation_space, action_space,
                                    replay_memory_size, history_length)
        #Episodic information
        self.episode_loss = 0
        self.episode_length = 0

        #Optimizer
        self.updater = mx.optimizer.get_updater(optimizer)

        #Target freezing
        self.freeze_interval = freeze_interval

        #Fix Params
        if self.params is None:
            self.params = self.qnet.params

        #DEBUG option
        self.debug_observation = False

    def act(self, observation):
        self.memory.append_obs(observation)
        if self.memory.size < self.memory.history_length:
            self.force_explore = True
        else:
            self.force_explore = False
        if self.force_explore or self.policy.decide_exploration():
            return self.action_space.sample()
        else:
            full_state = self.memory.latest_slice()
            norm_state = mx.nd.array(full_state.reshape((1,) + full_state.shape),
                                     ctx=self.ctx) / float(255.0)
            qval_npy = self.qnet.forward(is_train=False, data=norm_state)[0].asnumpy()
            action = np.argmax(qval_npy)
            return action

    def receive_feedback(self, reward, done):
        self.memory.add_feedback(self.current_action, reward,
                                 done)


        if self.is_learning.value:

            self.local_steps += 1
            if self.debug_observation and self.local_steps % 1000 == 0:
                cv2.imshow("observation", self.current_obs)
                cv2.waitKey(1)
            if self.local_steps > self.train_start \
                    and self.local_steps % self.training_interval == 0:
                self.train_once()

            if self.local_steps % self.freeze_interval == 0:
                self.qnet.copy_params_to(self.target_qnet)


            if done:
                logging.debug("l={:.4f},e={}".format(
                    self.episode_loss/self.episode_length,
                    self.policy.all_eps_current))
                self.policy.update_t(self.local_steps)
                self.episode_loss = 0
                self.episode_length = 0
    def compute_q_target(self, rewards, next_states, terminate_flags):
        target_qval = self.target_qnet.forward(is_train=False, data=next_states)[0]
        target_rewards = rewards + \
                         mx.nd.choose_element_0index(target_qval,
                                                     mx.nd.argmax_channel(target_qval))\
                         * (1.0 - terminate_flags) * self.discount
        return target_rewards

    def compute_double_q_target(self, rewards, next_states, terminate_flags):
        target_qval = self.target_qnet.forward(is_train=False, data=next_states)[0]
        qval = self.qnet.forward(is_train=False, data=next_states)[0]

        target_rewards = rewards + \
                         mx.nd.choose_element_0index(target_qval,
                                                     mx.nd.argmax_channel(qval)) \
                         * (1.0 - terminate_flags) * self.discount
        return target_rewards

    def train_once(self):
        states, actions, rewards, next_states, terminate_flags \
            = self.memory.sample(batch_size=self.minibatch_size)
        states = mx.nd.array(states, ctx=self.ctx) / float(255.0)
        next_states = mx.nd.array(next_states, ctx=self.ctx) / float(255.0)
        actions =mx.nd.array(actions, ctx=self.ctx)
        rewards = mx.nd.array(rewards, ctx=self.ctx)
        terminate_flags = mx.nd.array(terminate_flags, ctx=self.ctx)
        if self.is_double_q:
            target_rewards = self.compute_double_q_target(
                rewards, next_states, terminate_flags
            )
        else:
            target_rewards = self.compute_q_target(
                rewards, next_states, terminate_flags
            )
        q_outs = self.qnet.forward(
            is_train=True,
            data=states,
            dqn_action=actions,
            dqn_reward=target_rewards
        )
        self.qnet.backward()
        self.qnet.update(updater=self.updater)
        diff = nd.abs(nd.choose_element_0index(q_outs[0], actions) - target_rewards)
        quadratic_part = nd.clip(diff, -1, 1)
        loss = 0.5 * nd.sum(nd.square(quadratic_part)).asnumpy()[0] + \
               nd.sum(diff - quadratic_part).asnumpy()[0]

        # Episode recording
        self.episode_loss += loss
        self.episode_length += 1


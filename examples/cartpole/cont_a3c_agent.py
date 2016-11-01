import mxnet as mx
from arena import Base
from arena.agents import Agent
from arena.utils import discount_cumsum, norm_clipping
from mxnet.lr_scheduler import FactorScheduler
import numpy as np
import logging


class ContA3CAgent(Agent):
    def __init__(self, observation_space, action_space,
                 shared_params, stats_rx, acts_tx,
                 is_learning, global_t, pid=0,
                 lr=0.0001,
                 discount=0.99,
                 batch_size=4000,
                 ctx=mx.gpu(),
                 f_get_sym=None,
                 lr_scheduler=FactorScheduler(500, 0.1),
                 optimizer_name='adam',
                 clip_gradient=True
                 ):
        super(ContA3CAgent, self).__init__(
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid
        )
        self.action_dimension = action_space.low.shape[0]
        self.state_dimension = observation_space.low.shape[0]
        data_shapes = {'data': (batch_size, self.state_dimension),
                       'policy_score': (batch_size,),
                       'policy_backward_action': (batch_size, self.action_dimension),
                       'critic_label': (batch_size,)
                       }
        if f_get_sym is None:
            sym = self.actor_critic_policy_sym(self.action_dimension)
        else:
            sym = f_get_sym(self.action_dimension)
        if shared_params is None:
            net_param = None
        else:
            self.param_lock = shared_params["lock"]
            self.global_network = shared_params["global_net"]
            net_param = self.global_network.params
        self.net = Base(data_shapes=data_shapes, sym_gen=sym, name='ACNet',
                        params=net_param,
                        initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=1.0),
                        ctx=ctx)
        if shared_params is not None and "updater" in shared_params:
            self.updater = shared_params["updater"]
        else:
            if optimizer_name == 'sgd':
                optimizer = mx.optimizer.create(name='sgd', learning_rate=lr,
                                                lr_scheduler=lr_scheduler, momentum=0.9,
                                                clip_gradient=None, rescale_grad=1.0, wd=0.)
            elif optimizer_name == 'adam':
                optimizer = mx.optimizer.create(name='adam', learning_rate=lr,
                                                lr_scheduler=lr_scheduler)
            elif optimizer_name == 'adagrad':
                optimizer = mx.optimizer.create(name='adagrad', learning_rate=lr,
                                                eps=0.01,
                                                lr_scheduler=lr_scheduler)
            else:
                optimizer = mx.optimizer.create(name='sgd', learning_rate=lr,
                                                lr_scheduler=lr_scheduler, momentum=0.9,
                                                clip_gradient=None, rescale_grad=1.0, wd=0.)
            self.updater = mx.optimizer.get_updater(optimizer)

        # Constant Storage
        self.batch_size = batch_size
        self.discount = discount
        self.clip_gradient = clip_gradient

        # State information
        self.counter = batch_size
        self.episode_step = 0
        max_l = 500
        self.observation_buffer = np.empty(shape=(batch_size + max_l, self.state_dimension), dtype=np.float32)
        self.action_buffer = np.empty(shape=(batch_size + max_l, self.action_dimension), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(batch_size + max_l,), dtype=np.float32)
        self.value_buffer = np.empty(shape=(batch_size + max_l,), dtype=np.float32)
        self.td_buffer = np.empty(shape=(batch_size + max_l,), dtype=np.float32)
        self.buffer_size = 0
        self.buffer_episode_start = 0
        self.num_episodes = 0

    def act(self, observation):
        # if self.buffer_size == 0:
        #     self.global_network.copy_params_to(self.net)

        action = self.net.forward(is_train=False,
                                  data=observation.reshape(1, observation.size))[0].asnumpy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.flatten()
        return action

    def receive_feedback(self, reward, done):
        self.buffer_size += 1
        last_idx = self.buffer_size - 1
        self.observation_buffer[last_idx, :] = self.current_obs
        self.action_buffer[last_idx] = self.current_action
        self.reward_buffer[last_idx] = reward
        self.episode_step += 1
        if done:
            self.counter -= self.episode_step
            self.add_path()
            self.num_episodes += 1
            self.episode_step = 0
            self.buffer_episode_start = self.buffer_size
            if self.counter <= 0:
                self.train_once()
                self.counter = self.batch_size
                self.buffer_size = 0
                self.buffer_episode_start = 0
                self.num_episodes = 0

    def add_path(self):
        outputs = self.net.forward(
            is_train=False,
            data=self.observation_buffer[self.buffer_episode_start:self.buffer_size],
        )
        rewards = self.reward_buffer[self.buffer_episode_start:self.buffer_size]
        critics = outputs[3].asnumpy().reshape(rewards.shape)
        self.value_buffer[self.buffer_episode_start:self.buffer_size] = \
            discount_cumsum(rewards, self.discount)
        self.td_buffer[self.buffer_episode_start:self.buffer_size] = \
            self.value_buffer[self.buffer_episode_start:self.buffer_size] - critics


    def train_once(self):

        outputs = self.net.forward(
            is_train=True,
            data=self.observation_buffer[0:self.buffer_size],
        )
        scores = self.td_buffer[0:self.buffer_size]
        actions = self.action_buffer[0:self.buffer_size]
        values = self.value_buffer[0:self.buffer_size].flatten()
        self.net.backward(
            policy_score=scores,
            policy_backward_action=actions,
            critic_label=values,
                          )
        for grad in self.net.params_grad.values():
            grad[:] = grad[:] / self.buffer_size
        if self.clip_gradient:
            norm_clipping(self.net.params_grad, 10)
        with self.param_lock:
            # self.global_network.update(self.updater, params_grad=self.net.params_grad)
            self.net.update(self.updater, params_grad=self.net.params_grad)
        logging.info(
            'Thd[%d] Average Return:%f,  Num Traj:%d ' \
            % (self.id,
               np.sum(self.reward_buffer[0:self.buffer_size]) / self.num_episodes,
               self.num_episodes,
               ))


    def actor_critic_policy_sym(self, action_num):
        # define the network structure of Gaussian policy
        data = mx.symbol.Variable('data')
        policy_layers = mx.symbol.FullyConnected(data=data, name='fc_mean_1', num_hidden=128)
        policy_layers = mx.symbol.Activation(data=policy_layers, name='fc_mean_tanh_1', act_type='tanh')
        policy_layers = mx.symbol.FullyConnected(data=policy_layers, name='fc_mean_2', num_hidden=128)
        policy_layers = mx.symbol.Activation(data=policy_layers, name='fc_mean_tanh_2', act_type='tanh')
        policy_mean = mx.symbol.FullyConnected(data=policy_layers, name='fc_output', num_hidden=action_num)

        policy_var = mx.symbol.FullyConnected(data=policy_layers, name='fc_ovar', num_hidden=action_num)
        policy_var = mx.symbol.Activation(data=policy_var, name='fc_var', act_type='softrelu')

        # policy_var = mx.symbol.Variable('var')

        policy_net = mx.symbol.Custom(mean=policy_mean, var=policy_var,
                                      name='policy', op_type='LogNormalPolicy', implicit_backward=False)

        # define the network structure of critics network
        critic_net = mx.symbol.FullyConnected(data=data, name='fc_critic_1', num_hidden=128)
        critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_1', act_type='relu')
        critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_2', num_hidden=128)
        critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_2', act_type='relu')
        critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_3', num_hidden=1)
        target = mx.symbol.Variable('critic_label')
        critic_net = mx.symbol.LinearRegressionOutput(data=critic_net, name='critic', label=target, grad_scale=1)

        net = mx.symbol.Group([policy_net, mx.symbol.BlockGrad(policy_mean),
                               mx.symbol.BlockGrad(policy_var), critic_net])
        return net

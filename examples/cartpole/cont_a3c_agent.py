import mxnet as mx
from arena import Base
from arena.memory import AcMemory
from arena.agents import Agent
from arena.utils import discount_cumsum, norm_clipping
from arena.mp_utils import force_map
from mxnet.lr_scheduler import FactorScheduler
from mxnet import nd
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
            net_param = None
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
        self.epoch_reward = 0
        max_l = 500
        self.memory = AcMemory(observation_shape=(self.state_dimension,),
                               action_shape=(self.action_dimension,),
                               max_size=batch_size + max_l,
                               gamma=self.discount,
                               use_gae=False)

        self.num_episodes = 0

    def act(self, observation):
        # logging.debug("rx obs: {}".format(observation))
        if self.memory.Tmax == 0:
            self.global_network.copy_params_to(self.net)


        outputs = self.net.forward(is_train=False,
                                   data=np.expand_dims(observation, axis=0))
        action = \
            np.clip(outputs[0].asnumpy(), self.action_space.low, self.action_space.high).flatten()

        self.memory.append_state(observation, action, critic=outputs[3].asnumpy())


        # logging.debug("tx a: {}".format(action))


        return action

    def receive_feedback(self, reward, done):
        # logging.debug("rx r: {} \td:{}".format(reward, done))

        self.memory.append_feedback(reward)
        self.episode_step += 1
        self.epoch_reward += reward
        if done:
            self.counter -= self.episode_step
            # self.add_path()
            self.memory.add_path(done)
            self.num_episodes += 1
            self.episode_step = 0
            if self.counter <= 0:
                self.train_once()
                self.memory.reset()
                self.counter = self.batch_size
                self.num_episodes = 0


    def train_once(self):

        train_data = self.memory.extract_all()
        self.net.forward_backward(
            data=train_data['observations'],
            policy_score=train_data['advantages'],
            policy_backward_action=train_data['actions'],
            critic_label=train_data['values'],
        )

        def scale_gradient(grad):
            grad[:] /= self.memory.Tmax

        # force_map(scale_gradient, self.net.params_grad.values())
        for grad in self.net.params_grad.values():
            grad[:] = grad[:] / self.memory.Tmax
        if self.clip_gradient:
            norm_clipping(self.net.params_grad, 10)
        with self.param_lock:
            self.global_network.update(self.updater, params_grad=self.net.params_grad)
        logging.info(
            'Thd[%d] Average Return:%f,  Num Traj:%d ' \
            % (self.id,
               self.epoch_reward / self.num_episodes,
               self.num_episodes,
               ))
        self.epoch_reward = 0


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

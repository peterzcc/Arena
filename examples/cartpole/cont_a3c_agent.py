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
            shared_acnet = None
        else:
            shared_acnet = shared_params["acnet"]
            self.param_lock = shared_params["lock"]
        self.net = Base(data_shapes=data_shapes, sym_gen=sym, name='ACNet',
                        params=shared_acnet,
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
        self.observations = []
        self.actions = []
        self.rewards = []
        self.paths = []

    def act(self, observation):

        action = self.net.forward(is_train=False,
                                  data=observation.reshape(1, observation.size))[0].asnumpy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.flatten()
        return action

    def receive_feedback(self, reward, done):
        self.observations.append(self.current_obs)
        self.actions.append(self.current_action)
        self.rewards.append(reward)
        self.episode_step += 1
        if done:
            self.counter -= self.episode_step
            self.add_path()
            self.episode_step = 0
            if self.counter <= 0:
                self.train_once()
                self.counter = self.batch_size

    def add_path(self):
        observations = np.array(self.observations)
        rewards = np.array(self.rewards)
        outputs = self.net.forward(is_train=False,
                                   data=observations,
                                   )
        critics = outputs[3].asnumpy().reshape(rewards.shape)
        q_estimations = discount_cumsum(rewards, self.discount)
        advantages = q_estimations - critics
        path = dict(
            actions=np.array(self.actions),
            rewards=rewards,
            observations=observations,
            q_estimations=q_estimations,
            advantages=advantages,
        )
        self.paths.append(path)

        self.observations = []
        self.actions = []
        self.rewards = []

    def train_once(self):
        observations = np.concatenate([p["observations"] for p in self.paths])
        actions = np.concatenate([p["actions"] for p in self.paths])
        q_estimations = np.concatenate([p["q_estimations"] for p in self.paths])
        advantages = np.concatenate([p['advantages'] for p in self.paths])
        cur_batch_size = observations.shape[0]
        outputs = self.net.forward(is_train=True, data=observations,
                                   )
        policy_actions = outputs[0].asnumpy()
        critics = outputs[3].asnumpy()
        variance = outputs[2].asnumpy()
        action_mean = outputs[1].asnumpy()
        self.net.backward(policy_score=advantages,
                          policy_backward_action=actions,
                          critic_label=q_estimations.reshape(q_estimations.size, ),
                          )
        for grad in self.net.params_grad.values():
            grad[:] = grad[:] / cur_batch_size
        if self.clip_gradient:
            norm_clipping(self.net.params_grad, 10)
        with self.param_lock:
            self.net.update(self.updater)
        logging.info(
            'Average Return:%f, Max Return:%f, Min Return:%f, Num Traj:%d\n, Mean:%f, Var:%f, Average Baseline:%f' \
            % (np.mean([sum(p["rewards"]) for p in self.paths]),
               np.max([sum(p["rewards"]) for p in self.paths]),
               np.min([sum(p["rewards"]) for p in self.paths]),
               len(self.paths), action_mean.mean(), variance.mean(), critics.mean()
               ))

        self.paths = []

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

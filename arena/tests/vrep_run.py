__author__ = 'flyers'

from arena.games import VREPGame
from arena.utils import *
from arena.operators import *
from arena import Base
import numpy
import time
import logging
import sys
import mxnet as mx
import mxnet.ndarray as nd
import matplotlib.pyplot as plt


class NormalInitializer(mx.initializer.Xavier):
    def _init_weight(self, name, arr):
        super(NormalInitializer, self)._init_weight(name, arr)
        if name == 'fc_var_1_weight':
            arr[:] = 0

    def _init_bias(self, name, arr):
        super(NormalInitializer, self)._init_bias(name, arr)
        if name == 'fc_mean_3_bias':
            arr[:] = 0
        elif name == 'fc_var_1_bias':
            arr[:] = numpy.log(numpy.e - 1)

def regression_policy_sym(action_num):
    data = mx.symbol.Variable('data')
    net_mean = mx.symbol.FullyConnected(data=data, name='fc_mean_1', num_hidden=128)
    net_mean = mx.symbol.Activation(data=net_mean, name='fc_mean_relu_1', act_type='relu')
    net_mean = mx.symbol.FullyConnected(data=net_mean, name='fc_mean_2', num_hidden=64)
    net_mean = mx.symbol.Activation(data=net_mean, name='fc_mean_relu_2', act_type='relu')
    # net_mean = mx.symbol.FullyConnected(data=net_mean, name='fc_mean_3', num_hidden=64)
    # net_mean = mx.symbol.Activation(data=net_mean, name='fc_mean_relu_3', act_type='relu')
    net_mean = mx.symbol.FullyConnected(data=net_mean, name='output', num_hidden=action_num)
    target = mx.symbol.Variable('regression_label')
    net_mean = mx.symbol.LinearRegressionOutput(data=net_mean, label=target)
    # net_mean = output_op(data=net_mean, name='policy')
    return net_mean


def actor_critic_policy_sym(action_num):
    data = mx.symbol.Variable('data')
    policy_mean = mx.symbol.FullyConnected(data=data, name='fc_mean_1', num_hidden=128)
    policy_mean = mx.symbol.Activation(data=policy_mean, name='fc_mean_tanh_1', act_type='tanh')
    policy_mean = mx.symbol.FullyConnected(data=policy_mean, name='fc_mean_2', num_hidden=128)
    policy_mean = mx.symbol.Activation(data=policy_mean, name='fc_mean_tanh_2', act_type='tanh')
    policy_mean = mx.symbol.FullyConnected(data=policy_mean, name='fc_mean_3', num_hidden=action_num)
    policy_var = mx.symbol.Variable('var')
    # policy_var = mx.symbol.FullyConnected(data=data, name='fc_var_1', num_hidden=action_num)
    # policy_var = mx.symbol.Activation(data=policy_var, name='var_output', act_type='softrelu')
    policy_net = mx.symbol.Custom(mean=policy_mean, var=policy_var,
            name='policy', op_type='LogNormalPolicy', implicit_backward=False, deterministic=True)

    critic_net = mx.symbol.FullyConnected(data=data, name='fc_critic_1', num_hidden=128)
    critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_1', act_type='relu')
    critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_2', num_hidden=128)
    critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_2', act_type='relu')
    critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_3', num_hidden=1)
    target = mx.symbol.Variable('critic_label')
    critic_net = mx.symbol.LinearRegressionOutput(data=critic_net, name='critic', label=target, grad_scale=1)

    net = mx.symbol.Group([policy_net, critic_net])
    return net


root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

replay_start_size = 1000
replay_memory_size = 1000000
history_length = 10
frame_skip = 1

game = VREPGame(replay_start_size=replay_start_size, frame_skip=frame_skip, remote_port=19997,
                history_length=history_length, replay_memory_size=replay_memory_size)
npy_rng = get_numpy_rng()
action_dim = game.replay_memory.action_dim
state_dim = game.replay_memory.state_dim

ctx = mx.gpu()
n_itr = 1000
episode_T = 500
batch_size = 1000

discount = 0.99
learning_rate = 0.0005

data_shapes = {'data': (episode_T, state_dim[0] * history_length),
               'policy_backward_action': (episode_T, action_dim[0]),
               'policy_score': (episode_T, ),
               'critic_label': (episode_T, 1),
               'var': (episode_T, action_dim[0]),
               }

sym = actor_critic_policy_sym(action_dim[0])
net = Base(data_shapes=data_shapes, sym=sym, name='ACNet-vrep',
           initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=1.0), ctx=ctx)
net.load_params(name=net.name, dir_path='./models', epoch=380)

# regression_net = regression_policy_sym(action_dim[0])
# regression_data_shapes = {'data': (episode_T, game.replay_memory.state_dim[0] * history_length),
#                'regression_label': (episode_T, action_dim[0])}
# regression_net = Base(data_shapes=regression_data_shapes, sym=regression_net, name='PolicyNet-demo',
#            initializer=mx.initializer.Xavier(factor_type='in', magnitude=1.0), ctx=ctx)
# regression_net.load_params(name='PolicyNet-demo', dir_path='./models/', epoch=999)
# net.params['fc_mean_1_weight'][:] = regression_net.params['fc_mean_1_weight']
# net.params['fc_mean_1_bias'][:] = regression_net.params['fc_mean_1_bias']
# net.params['fc_mean_2_weight'][:] = regression_net.params['fc_mean_2_weight']
# net.params['fc_mean_2_bias'][:] = regression_net.params['fc_mean_2_bias']
# net.params['fc_mean_3_weight'][:] = regression_net.params['output_weight']
# net.params['fc_mean_3_bias'][:] = regression_net.params['output_bias']
# print net.params['fc_mean_3_bias'].asnumpy()

# optimizer = mx.optimizer.create(name='sgd', learning_rate=0.000001,
#                                 clip_gradient=None, rescale_grad=1.0, wd=0.)
optimizer = mx.optimizer.create(name='adam', learning_rate=learning_rate)
updater = mx.optimizer.get_updater(optimizer)

net.print_stat()

# Begin playing game
for itr in xrange(n_itr):
    paths = []
    counter = batch_size
    N = 0
    for _ in xrange(1):
        N += 1
        observations = []
        actions = []
        rewards = []
        game.begin_episode(episode_T)
        for step in xrange(episode_T):
            observation = game.current_state()
            action = net.forward(batch_size=1, is_train=False,
                                 data=observation.reshape(1, -1),
                                 var=1.*numpy.ones((1,action_dim[0])),
                                 )[0].asnumpy()
            action = action.reshape(action.size)
            reward, terminate = game.play(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            if terminate:
                break

        path = dict(
            actions=numpy.array(actions),
            observations=numpy.array(observations).reshape(step+1, -1),
            rewards=numpy.array(rewards),
        )
        counter -= (step+1)
        return_so_far = 0
        # if terminate:
        #     return_so_far = 0
        # else:
        #     observation = game.current_state()
        #     outputs = net.forward(batch_size=1, is_train=False,
        #                           data=numpy.array(observation).reshape(1, -1),
        #                           var=1.*numpy.ones((1,)),
        #                           )
        #     return_so_far = outputs[1].asnumpy()[0, 0]
        outputs = net.forward(batch_size=path['observations'].shape[0], is_train=False,
                              data=path['observations'],
                              var=1.*numpy.ones((path['observations'].shape[0],action_dim[0]))
                              )
        critics = outputs[3].asnumpy().reshape(path['rewards'].shape)
        returns = []
        advantages = []
        for t in xrange(len(rewards) - 1, -1, -1):
            return_so_far = rewards[t] + discount * return_so_far
            returns.append(return_so_far)
            advantage = return_so_far - critics[t]
            advantages.append(advantage)
        path['returns'] = numpy.array(returns[::-1])
        path['advantages'] = numpy.array(advantages[::-1])
        paths.append(path)

    observations = numpy.concatenate([p["observations"] for p in paths])
    actions = numpy.concatenate([p["actions"] for p in paths])
    returns = numpy.concatenate([p["returns"] for p in paths])
    advantages = numpy.concatenate([p['advantages'] for p in paths])
    cur_batch_size = observations.shape[0]
    outputs = net.forward(batch_size=cur_batch_size, is_train=True, data=observations,
                          var=1.*numpy.ones((cur_batch_size,action_dim[0])),
                          )
    policy_actions = outputs[0].asnumpy()
    action_mean = outputs[1].asnumpy()
    action_var = outputs[2].asnumpy()
    critics = outputs[3].asnumpy()
    # net.backward(batch_size=cur_batch_size,
    #              policy_score=advantages,
    #              policy_backward_action=actions,
    #              critic_label=returns.reshape(returns.size, 1),
    #              )
    # net.update(updater)
    # for ind, k in enumerate(net.params.keys()):
    #     updater(index=ind, grad=net.params_grad[k]/cur_batch_size, weight=net.params[k])
    logging.info('Epoch:%d, Batchsize:%d, Average Return:%f, Estimated Baseline:%f, Num Traj:%d, Var:%f, Mean:%f' %(itr, cur_batch_size, numpy.mean([sum(p["rewards"]) for p in paths]), critics.mean(), N, action_var.mean(), action_mean.mean()))


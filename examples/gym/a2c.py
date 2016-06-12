import numpy as np
import mxnet as mx
from arena.operators import *
from mxnet.lr_scheduler import FactorScheduler
from arena.utils import *
from arena import Base
import argparse
import gym

def normal_sym(action_num):
    # define the network structure of Gaussian policy
    data = mx.symbol.Variable('data')
    policy_mean = mx.symbol.FullyConnected(data=data, name='fc_mean_1', num_hidden=128)
    policy_mean = mx.symbol.Activation(data=policy_mean, name='fc_mean_tanh_1', act_type='tanh')
    policy_mean = mx.symbol.FullyConnected(data=policy_mean, name='fc_mean_2', num_hidden=128)
    policy_mean = mx.symbol.Activation(data=policy_mean, name='fc_mean_tanh_2', act_type='tanh')
    policy_mean = mx.symbol.FullyConnected(data=policy_mean, name='fc_output', num_hidden=action_num)

    # policy_var = mx.symbol.Variable('var')
    policy_var = mx.symbol.FullyConnected(data=data, name='fc_var', num_hidden=action_num)
    policy_var = mx.symbol.exp(data=policy_var, name='var_out')

    policy_net = mx.symbol.Custom(mean=policy_mean, var=policy_var,
            name='policy', op_type='LogNormalPolicy', implicit_backward=False)
    return policy_net

def softmax_sym(action_num):
    data = mx.symbol.Variable('data')
    net = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
    net = mx.symbol.Activation(data=net, name='relu1', act_type='relu')
    net = mx.symbol.FullyConnected(data=net, name='fc2', num_hidden=128)
    net = mx.symbol.Activation(data=net, name='relu2', act_type='relu')
    net = mx.symbol.FullyConnected(data=net, name='fc_out', num_hidden=action_num)

    net = mx.symbol.Custom(data=net, name='policy', op_type='LogSoftmaxPolicy', implicit_backward=False)
    return net


def critic_sym():
    data = mx.symbol.Variable('data')
    # define the network structure of critics network
    critic_net = mx.symbol.FullyConnected(data=data, name='fc_critic_1', num_hidden=128)
    critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_1', act_type='relu')
    critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_2', num_hidden=128)
    critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_2', act_type='relu')
    critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_3', num_hidden=1)
    target = mx.symbol.Variable('critic_label')
    critic_net = mx.symbol.LinearRegressionOutput(data=critic_net, name='critic', label=target, grad_scale=1)

    return critic_net


parser = argparse.ArgumentParser(description='Script to train controllers in the gym environment by advantage actor-critic algorithm.')
parser.add_argument('--lr', required=False, type=float, help='learning rate of the choosen optimizer')
parser.add_argument('--optimizer', default='sgd', type=str, help='choice of the optimizer, adam or sgd')
parser.add_argument('--clip-gradient', default=True, type=str, help='whether to clip the gradient')
parser.add_argument('--save-model', default=False, type=str, help='whether to save the final model')
args, unknow = parser.parse_known_args()

if args.lr is None:
    args.lr = 0.005

# Each trajectory will have at most 500 time steps
T = 500
# Set the discount factor for the problem
discount = 0.99
n_itr = 500
batch_size = 4000
ctx = mx.gpu()

env = gym.make('CartPole-v0')
observation_shape = env.observation_space.shape
if 'Discrete' in env.action_space.__repr__():
    action_shape = (env.action_space.n, )
    policy_data_shapes = {'data': (T,) + observation_shape,
                          'policy_score': (T,),
                          'policy_backward_action': (T,),
                          }
    policy_sym = softmax_sym(action_shape[0])
elif 'Box' in env.action_space.__repr__():
    action_shape = env.action_space.shape
    policy_data_shapes = {'data': (T,) + observation_shape,
                          'policy_score': (T,),
                          'policy_backward_action': (T,) + action_shape,
                          }
    policy_sym = normal_sym(action_shape[0])
else:
    raise ValueError('Unrecognized action space type')
policy_net = Base(data_shapes=policy_data_shapes, sym=policy_sym, name='a2c-a',
                  initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=1.0), ctx=ctx)

critic_data_shapes = {'data': (T,) + observation_shape,
                      'critic_label':(T, 1),
                      }
critic_sym = critic_sym()
critic_net = Base(data_shapes=critic_data_shapes, sym=critic_sym, name='a2c-c',
                  initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=1.0), ctx=ctx)

lr_scheduler = FactorScheduler(500, 0.1)
if args.optimizer == 'sgd':
    optimizer = mx.optimizer.create(name='sgd', learning_rate=args.lr,
                                    lr_scheduler=lr_scheduler, momentum=0.9,
                                    clip_gradient=None, rescale_grad=1.0, wd=0.)
elif args.optimizer == 'adam':
    optimizer = mx.optimizer.create(name='adam', learning_rate=args.lr,
                                    lr_scheduler=lr_scheduler)
else:
    raise ValueError('optimizer must be chosen between adam and sgd')
updater1 = mx.optimizer.get_updater(optimizer)
updater2 = mx.optimizer.get_updater(optimizer)


for itr in xrange(n_itr):
    paths = []
    counter = batch_size
    N = 0
    while counter > 0:
        N += 1
        observations = []
        actions = []
        rewards = []

        observation = env.reset()
        for step in xrange(T):
            action = policy_net.forward(batch_size=1, is_train=False,
                                        data=observation.reshape(1, observation.size),
                                        )[0].asnumpy()
            next_observation, reward, terminal, _ = env.step(action[0])
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            observation = next_observation
            if terminal:
                break

        counter -= (step + 1)
        observations = np.array(observations)
        rewards = np.array(rewards)
        critic_outputs = critic_net.forward(batch_size=observations.shape[0], is_train=False,
                                            data=observations,)
        critics = critic_outputs[0].asnumpy().reshape(rewards.shape)
        q_estimations = discount_cumsum(rewards, discount)
        advantages = q_estimations - critics
        path = dict(
            actions=np.array(actions),
            rewards=rewards,
            observations=observations,
            q_estimations=q_estimations,
            advantages=advantages,
        )
        paths.append(path)

    observations = np.concatenate([p["observations"] for p in paths])
    actions = np.concatenate([p["actions"] for p in paths])
    q_estimations = np.concatenate([p["q_estimations"] for p in paths])
    advantages = np.concatenate([p['advantages'] for p in paths])
    cur_batch_size = observations.shape[0]
    policy_outputs = policy_net.forward(batch_size=cur_batch_size, is_train=True,
                                        data=observations,)
    critic_outputs = critic_net.forward(batch_size=cur_batch_size, is_train=True,
                                        data=observations)
    policy_actions = policy_outputs[0].asnumpy()
    critics = critic_outputs[0].asnumpy()
    policy_net.backward(batch_size=cur_batch_size, policy_score=advantages,
                        policy_backward_action=actions.reshape((actions.size,)))
    critic_net.backward(batch_size=cur_batch_size,
                        critic_label=q_estimations.reshape(q_estimations.size, 1))
    for grad in policy_net.params_grad.values():
        grad[:] = grad[:] / cur_batch_size
    for grad in critic_net.params_grad.values():
        grad[:] = grad[:] / cur_batch_size
    if args.clip_gradient:
        norm_clipping(policy_net.params_grad, 10)
        norm_clipping(critic_net.params_grad, 10)
    policy_net.update(updater1)
    critic_net.update(updater2)
    print 'Epoch:%d, Average Return:%f, Max Return:%f, Min Return:%f, Num Traj:%d, Average Baseline:%f' \
          %(itr, np.mean([sum(p["rewards"]) for p in paths]),
            np.max([sum(p["rewards"]) for p in paths]),
            np.min([sum(p["rewards"]) for p in paths]),
            N, critics.mean()
            )

if args.save_model:
    policy_net.save_params(dir_path='./', epoch=itr)


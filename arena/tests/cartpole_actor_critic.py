import numpy as np
import mxnet as mx
from arena.operators import *
from arena.utils import *
from arena import Base
from arena.games.cartpole_box2d import CartpoleSwingupEnv
import argparse

def actor_critic_policy_sym(action_num):
    data = mx.symbol.Variable('data')
    policy_mean = mx.symbol.FullyConnected(data=data, name='fc_mean_1', num_hidden=128)
    policy_mean = mx.symbol.Activation(data=policy_mean, name='fc_mean_tanh_1', act_type='tanh')
    policy_mean = mx.symbol.FullyConnected(data=policy_mean, name='fc_mean_2', num_hidden=128)
    policy_mean = mx.symbol.Activation(data=policy_mean, name='fc_mean_tanh_2', act_type='tanh')
    policy_mean = mx.symbol.FullyConnected(data=policy_mean, name='fc_output', num_hidden=action_dimension)
    # policy_mean = mx.symbol.Activation(data=policy_mean, name='fc_output_tanh', act_type='tanh')
    policy_var = mx.symbol.Variable('var')
    # policy_var = mx.symbol.FullyConnected(data=data, name='fc_var_1', num_hidden=action_num)
    # policy_var = mx.symbol.Activation(data=policy_var, name='var_output', act_type='softrelu')
    # policy_var = mx.symbol.exp(data=policy_var, name='var_output')
    # op = LogNormalPolicy(implicit_backward=False)
    # policy_net = op(mean=policy_mean, var=policy_var, name='policy')
    policy_net = mx.symbol.Custom(mean=policy_mean, var=policy_var,
            name='policy', op_type='LogNormalPolicy', implicit_backward=False)

    critic_net = mx.symbol.FullyConnected(data=data, name='fc_critic_1', num_hidden=128)
    critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_1', act_type='relu')
    critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_2', num_hidden=128)
    critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_2', act_type='relu')
    critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_3', num_hidden=1)
    target = mx.symbol.Variable('critic_label')
    critic_net = mx.symbol.LinearRegressionOutput(data=critic_net, name='critic', label=target, grad_scale=1)

    net = mx.symbol.Group([policy_net, critic_net])
    return net


parser = argparse.ArgumentParser(description='Script to test the network on cartpole swingup.')
parser.add_argument('--lr', required=False, type=float, help='learning rate of the choosen optimizer')
parser.add_argument('--opt', required=True, type=str, help='choice of the optimizer, adam or sgd')

args, unknow = parser.parse_known_args()
if args.lr == None:
    if args.opt == 'adam':
        args.lr = 0.001
    elif args.opt == 'sgd':
        args.lr = 0.0001

# Each trajectory will have at most 500 time steps
T = 500
# Number of iterations
n_itr = 200
batch_size = 4000
# Set the discount factor for the problem
discount = 0.99
# Learning rate for the gradient update
learning_rate = args.lr

ctx = mx.gpu()

action_dimension = 1
state_dimension = 4


env = CartpoleSwingupEnv()

data_shapes = {'data': (T, state_dimension),
               'policy_score': (T, ),
               'policy_backward_action': (T, action_dimension),
               'critic_label': (T, 1),
               'var': (T, action_dimension),
               }
sym = actor_critic_policy_sym(action_dimension)

net = Base(data_shapes=data_shapes, sym=sym, name='ACNet',
           initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=1.0), ctx=ctx)
if args.opt == 'sgd':
    optimizer = mx.optimizer.create(name='sgd', learning_rate=learning_rate,
                                    clip_gradient=None, rescale_grad=1.0, wd=0.)
elif args.opt == 'adam':
    optimizer = mx.optimizer.create(name='adam', learning_rate=learning_rate)
updater = mx.optimizer.get_updater(optimizer)

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
            action = net.forward(batch_size=1, is_train=False,
                                 data=observation.reshape(1, observation.size),
                                 var=1.*np.ones((1,1)),
                                 )[0].asnumpy()
            action = action.flatten()
            next_observation, reward, terminal, _ = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            observation = next_observation
            if terminal:
                break

        counter -= (step + 1)
        observations = np.array(observations)
        rewards = np.array(rewards)
        outputs = net.forward(batch_size=observations.shape[0], is_train=False,
                              data=observations,
                              var=1.*np.ones((observations.shape[0],1)),
                              )
        critics = outputs[3].asnumpy().reshape(rewards.shape)
        q_estimations = discount_cumsum(rewards, discount)
        advantages = q_estimations - critics
        path = dict(
            actions=np.array(actions),
            rewards=rewards,
            observations=observations,
            q_estimations = q_estimations,
            advantages = advantages,
        )
        paths.append(path)

    observations = np.concatenate([p["observations"] for p in paths])
    actions = np.concatenate([p["actions"] for p in paths])
    q_estimations = np.concatenate([p["q_estimations"] for p in paths])
    advantages = np.concatenate([p['advantages'] for p in paths])
    cur_batch_size = observations.shape[0]
    outputs = net.forward(batch_size=cur_batch_size, is_train=True, data=observations,
                          var=1.*np.ones((cur_batch_size,1)),
                          )
    policy_actions = outputs[0].asnumpy()
    critics = outputs[3].asnumpy()
    variance = outputs[2].asnumpy()
    action_mean = outputs[1].asnumpy()
    net.backward(batch_size=cur_batch_size,
                 policy_score=advantages,
                 policy_backward_action=actions,
                 critic_label=q_estimations.reshape(q_estimations.size, 1),
                 )
    for grad in net.params_grad.values():
        grad[:] = grad[:] / cur_batch_size
    norm_clipping(net.params_grad, 5)
    net.update(updater)
    print 'Epoch:%d, Batchsize:%d, Average Return:%f, Max Return:%f, Min Return:%f' %(itr, cur_batch_size, np.mean([sum(p["rewards"]) for p in paths]), np.max([sum(p["rewards"]) for p in paths]), np.min([sum(p["rewards"]) for p in paths]))
    print 'Epoch:%d, Estimated Baseline:%f, Num Traj:%d, Variance:%f, Mean:%f' %(itr, critics.mean(), N, variance.mean(), action_mean.mean())

#net.save_params(dir_path='./models/ac2/', epoch=itr)


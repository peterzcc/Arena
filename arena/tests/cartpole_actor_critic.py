from arena.games.cartpole_box2d import CartpoleSwingupEnv
import numpy as np
import mxnet as mx
from arena.utils import *
from arena.operators import *
from arena import Base

class NormalInitializer(mx.initializer.Xavier):
    def _init_weight(self, name, arr):
        super(NormalInitializer, self)._init_weight(name, arr)
    #     if name == 'fc_var_1_weight':
    #         arr[:] = 0

    def _init_bias(self, name, arr):
        super(NormalInitializer, self)._init_bias(name, arr)
    #     if name == 'fc_var_1_bias':
    #         arr[:] = 0.5
            # arr[:] = np.log(np.e - 1)


def actor_critic_policy_sym(action_num):
    data = mx.symbol.Variable('data')
    policy_mean = mx.symbol.FullyConnected(data=data, name='fc_mean_1', num_hidden=128)
    policy_mean = mx.symbol.Activation(data=policy_mean, name='fc_mean_tanh_1', act_type='tanh')
    policy_mean = mx.symbol.FullyConnected(data=policy_mean, name='fc_mean_2', num_hidden=128)
    policy_mean = mx.symbol.Activation(data=policy_mean, name='fc_mean_tanh_2', act_type='tanh')
    target = mx.symbol.Variable('critic_label')
    critic_net = mx.symbol.FullyConnected(data=policy_mean, name='fc_critic_2', num_hidden=128)
    critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_2', act_type='relu')
    critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_3', num_hidden=1)
    critic_net = mx.symbol.LinearRegressionOutput(data=critic_net, name='critic', label=target, grad_scale=1)
    policy_mean = mx.symbol.FullyConnected(data=policy_mean, name='fc_output', num_hidden=action_dimension)
    # policy_mean = 2*mx.symbol.Activation(data=policy_mean, name='fc_output_tanh', act_type='tanh')
    policy_var = mx.symbol.Variable('var')
    # policy_var = mx.symbol.FullyConnected(data=data, name='fc_var_1', num_hidden=action_num)
    # policy_var = mx.symbol.Activation(data=policy_var, name='var_output', act_type='softrelu')
    # policy_var = mx.symbol.exp(data=policy_var, name='var_output')
    # op = LogNormalPolicy(implicit_backward=False)
    # policy_net = op(mean=policy_mean, var=policy_var, name='policy')
    policy_net = mx.symbol.Custom(mean=policy_mean, var=policy_var,
            name='policy', op_type='LogNormalPolicy', implicit_backward=False)

    # critic_net = mx.symbol.FullyConnected(data=data, name='fc_critic_1', num_hidden=128)
    # critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_1', act_type='relu')
    # critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_2', num_hidden=128)
    # critic_net = mx.symbol.Activation(data=critic_net, name='fc_critic_relu_2', act_type='relu')
    # critic_net = mx.symbol.FullyConnected(data=critic_net, name='fc_critic_3', num_hidden=1)
    # target = mx.symbol.Variable('critic_label')
    # critic_net = mx.symbol.LinearRegressionOutput(data=critic_net, name='critic', label=target, grad_scale=1)

    net = mx.symbol.Group([policy_net, mx.symbol.BlockGrad(policy_mean), mx.symbol.BlockGrad(policy_var), critic_net])
    return net



# Each trajectory will have at most 500 time steps
T = 500
# Number of iterations
n_itr = 1000
batch_size = 4000
# Set the discount factor for the problem
discount = 0.99
# Learning rate for the gradient update
learning_rate = 0.0005

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
           initializer=NormalInitializer(rnd_type='gaussian', factor_type='avg', magnitude=1.0), ctx=mx.gpu())
# optimizer = mx.optimizer.create(name='sgd', learning_rate=0.0001,
#                                 clip_gradient=None, rescale_grad=1.0, wd=0.)
optimizer = mx.optimizer.create(name='adam', learning_rate=learning_rate)
updater = mx.optimizer.get_updater(optimizer)

# net.load_params(name=net.name, dir_path='./models/good', epoch=n_itr-1)

baseline = 0
for itr in xrange(n_itr):

    paths = []

    counter = batch_size
    N = 0
    while counter > 0:
    # for _ in xrange(1):
        N += 1
        observations = []
        actions = []
        rewards = []

        observation = env.reset()
        # print 'initial observation:', observation
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

        path_length = step + 1
        counter -= path_length
        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
        )

        returns = []
        advantages = []
        # if terminal:
        return_so_far = 0
        # else:
        #     outputs = net.forward(batch_size=1, is_train=False,
        #                           data=np.array(observation).reshape(1, -1))
        #     return_so_far = outputs[3].asnumpy()[0, 0]
        outputs =net.forward(batch_size=path['observations'].shape[0], is_train=False,
                             data=path['observations'],
                             var=1.*np.ones((path['observations'].shape[0],1)),
                             )
        critics = outputs[3].asnumpy().reshape(path['rewards'].shape)
        for t in xrange(len(rewards) - 1, -1, -1):
            return_so_far = rewards[t] + discount * return_so_far
            returns.append(return_so_far)
            advantage = return_so_far - critics[t]
            advantages.append(advantage)

        path['returns'] = np.array(returns[::-1])
        path['advantages'] = np.array(advantages[::-1])
        paths.append(path)

    observations = np.concatenate([p["observations"] for p in paths])
    actions = np.concatenate([p["actions"] for p in paths])
    returns = np.concatenate([p["returns"] for p in paths])
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
                 critic_label=returns.reshape(returns.size, 1),
                 )
    # net.update(updater)
    for ind, k in enumerate(net.params.keys()):
        updater(index=ind, grad=net.params_grad[k]/cur_batch_size, weight=net.params[k])
    #print 'Epoch:%d, Average Return:%f' %(itr, np.mean([sum(p["rewards"]) for p in paths]))
    print 'Epoch:%d, Batchsize:%d, Average Return:%f, Estimated Baseline:%f, Num Traj:%d, Variance:%f, Mean:%f' %(itr, cur_batch_size, np.mean([sum(p["rewards"]) for p in paths]), critics.mean(), N, variance.mean(), action_mean.mean())

net.save_params(dir_path='./models/ac2/', epoch=itr)


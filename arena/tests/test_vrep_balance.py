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
    def _init_bias(self, name, arr):
        if name == 'fc_mean_3_bias':
            arr[:] = 5.335
        elif name == 'fc_var_1_bias':
            arr[:] = -3
        else:
            arr[:] = 0

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

def normal_policy_sym(action_num, output_op):
    data = mx.symbol.Variable('data')
    net_mean = mx.symbol.FullyConnected(data=data, name='fc_mean_1', num_hidden=128)
    net_mean = mx.symbol.Activation(data=net_mean, name='fc_mean_relu_1', act_type='relu')
    net_mean = mx.symbol.FullyConnected(data=net_mean, name='fc_mean_2', num_hidden=64)
    net_mean = mx.symbol.Activation(data=net_mean, name='fc_mean_relu_2', act_type='relu')
    net_mean = mx.symbol.FullyConnected(data=net_mean, name='fc_mean_3', num_hidden=action_num)
    net_var = mx.symbol.FullyConnected(data=data, name='fc_var_1', num_hidden=action_num)
    net_var = mx.symbol.Activation(data=net_var, name='fc_var_softplus_1', act_type='softrelu')
    net_out = output_op(mean=net_mean, var=net_var, name='policy')
    return net_mean, net_var, net_out


root = logging.getLogger()
root.setLevel(logging.DEBUG)

# ch = logging.StreamHandler(sys.stdout)
ch = logging.FileHandler('vrep.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

replay_start_size = 1000
replay_memory_size = 1000000
history_length = 15
frame_skip = 1

game = VREPGame(replay_start_size=replay_start_size, frame_skip=frame_skip,remote_port=20000,
                history_length=history_length, replay_memory_size=replay_memory_size)
npy_rng = get_numpy_rng()
action_dim = game.replay_memory.action_dim

ctx = mx.gpu()
epoch_num = 100
steps_per_epoch = 100000
update_interval = 1
reward_discount = 0.99

episode_T = 128
data_shapes = {'data': (episode_T, game.replay_memory.state_dim[0] * history_length),
               'policy_backward_action': (episode_T, action_dim[0]),
               'policy_score': (episode_T, )}
# data_shapes = {'data': (minibatch_size, history_length) + game.replay_memory.state_dim,
#                'policy_reward': (minibatch_size, )}


normal_output_op = LogNormalPolicy(implicit_backward=False)
policy_net_mean, policy_net_var, policy_net = normal_policy_sym(
    game.replay_memory.action_dim[0], normal_output_op)

net = Base(data_shapes=data_shapes, sym=policy_net, name='PolicyNet-balance',
           initializer=NormalInitializer(rnd_type='gaussian', factor_type='in', magnitude=1.0), ctx=ctx)

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

optimizer = mx.optimizer.create(name='sgd', learning_rate=0.000001,
                                clip_gradient=None, rescale_grad=1.0, wd=0.)
updater = mx.optimizer.get_updater(optimizer)

net.print_stat()

# Begin playing game
training_steps = 0
total_steps = 0
baseline = 0
for epoch in xrange(epoch_num):
    # Run epoch
    steps_left = steps_per_epoch
    episode = 0
    epoch_reward = 0
    start_epoch = time.time()
    while steps_left > 0:
        # Running new episode
        episode += 1
        episode_loss = 0.0
        episode_update_step = 0
        game.begin_episode(steps_left)
        time_episode_start = time.time()
        current_state = game.current_state()
        initial_state = current_state[history_length-1].reshape(1, current_state.shape[1])
        while not game.episode_terminate():
            current_state = game.current_state().flatten()
            state = current_state.reshape((1, current_state.size))
            policy_means = net.forward(sym_name="fc_mean_3_output", batch_size=1,
                                       is_train=False, data=state)[0].asnumpy()
            policy_vars = net.forward(sym_name='fc_var_softplus_1_output',
                                      batch_size=1,
                                      is_train=False, data=state)[0].asnumpy()
            action = net.forward(is_train=False, batch_size=1,
                                 data=state)[0].asnumpy()
            action = action.reshape((action.shape[1], ))
            logging.info('played action')
            logging.info(action)

            # play the game
            game.play(action)
            total_steps += 1

        # update the policy network
        # draw sample from the replay memory
        training_steps += 1
        episode_update_step += 1

        states, actions, rewards, terminate_flag, batch_size = \
            game.replay_memory.latest_episode(initial_state=initial_state)
        assert terminate_flag
        data_batch = numpy.zeros((batch_size, data_shapes['data'][1]))
        if batch_size-history_length+1 >= 0:
            end_state = states[numpy.arange(batch_size-history_length+1, batch_size+1), :].flatten()
        else:
            end_state = numpy.tile(states[batch_size], history_length)
        for i in range(min(history_length-1, batch_size)):
            data_batch[i] = numpy.tile(states[i], history_length)
        for i in range(history_length-1, batch_size):
            data_batch[i] = states[numpy.arange(i-history_length+1, i+1), :].flatten()
        reward_so_far = 0
        for t in xrange(rewards.size-1, -1, -1):
            reward_so_far = reward_discount * reward_so_far + rewards[t]
            rewards[t] = reward_so_far
        baseline -= 0.001 * (baseline - rewards.mean(axis=0))
        policy_means = net.forward(sym_name='fc_mean_3_output',
                                   batch_size=batch_size,
                                   is_train=False, data=data_batch)[0].asnumpy()
        policy_vars = net.forward(sym_name='fc_var_softplus_1_output',
                                  batch_size=batch_size, is_train=False,
                                  data=data_batch)[0].asnumpy()
        policy_actions = net.forward(is_train=True, batch_size=batch_size,
                                     data=data_batch)[0].asnumpy()
        net.backward(batch_size=batch_size,
                     policy_score=rewards - baseline,
                     policy_backward_action=actions)
        net.update(updater)

        cur_loss = -numpy.log(numpy.sqrt(policy_vars)) + 0.5*numpy.square(policy_means-policy_actions) / policy_vars
        loss = numpy.sum(numpy.sum(cur_loss, axis=1) * (rewards - baseline)) / rewards.size

        logging.info('baseline:%f, policy mean:%f, policy var:%f\n' % (baseline, policy_means.mean(), policy_vars.mean()))
        episode_loss += loss

        time_episode_end = time.time()
        steps_left -= game.episode_step
        # update the statistics
        epoch_reward += game.episode_reward
        info_str = 'Epoch:%d, Episode:%d, Steps Left:%d/%d, Reward:%f, fps:%f' \
                   % (epoch, episode, steps_left, steps_per_epoch, game.episode_reward, game.episode_step / (time_episode_end - time_episode_start))
        if episode_update_step > 0:
            info_str += ', Avg Loss:%f/%d' % (episode_loss / episode_update_step,
                                            episode_update_step)
        logging.info(info_str)

    end_epoch = time.time()
    net.save_params('./models/', epoch=epoch)
    logging.info('Epoch:%d, FPS:%f, Avg Reward: %f/%d' \
          % (epoch, steps_per_epoch / (end_epoch-start_epoch), epoch_reward / float(episode), episode))



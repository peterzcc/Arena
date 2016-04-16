__author__ = 'flyers'

from arena.games import VREPGame
from arena.utils import *
from arena.operators import *
from arena import Base
import numpy
import time
import mxnet as mx
import mxnet.ndarray as nd
import matplotlib.pyplot as plt

def normal_policy_sym(action_num, output_op):
    data = mx.symbol.Variable('data')
    net_mean = mx.symbol.FullyConnected(data=data, name='fc_mean_1', num_hidden=64)
    net_mean = mx.symbol.Activation(data=net_mean, name='fc_mean_relu_1', act_type='relu')
    net_mean = mx.symbol.FullyConnected(data=net_mean, name='fc_mean_2', num_hidden=32)
    net_mean = mx.symbol.Activation(data=net_mean, name='fc_mean_relu_2', act_type='relu')
    net_mean = mx.symbol.FullyConnected(data=net_mean, name='fc_mean_3', num_hidden=action_num)
    net_var = mx.symbol.FullyConnected(data=data, name='fc_var_1', num_hidden=action_num)
    net_var = mx.symbol.Activation(data=net_var, name='fc_var_softplus_1', act_type='softrelu')
    net_out = output_op(mean=net_mean, var=net_var, name='policy')
    return net_mean, net_var, net_out

replay_start_size = 1000
replay_memory_size = 1000000
history_length = 1
frame_skip = 1
episode_T = 32

game = VREPGame(replay_start_size=replay_start_size, frame_skip=frame_skip,
                history_length=history_length, replay_memory_size=replay_memory_size)
npy_rng = get_numpy_rng()
action_dim = game.replay_memory.action_dim

ctx = mx.gpu()
epoch_num = 100
steps_per_epoch = 10000
update_interval = 1
reward_discount = 1

# eps_start = 0.5
eps_start = 1.0
eps_min = 0.1
eps_decay = (eps_start - eps_min) / 100000
eps_curr = eps_start
minibatch_size = 64
data_shapes = {'data': (minibatch_size, ) + game.replay_memory.state_dim,
               'policy_score': (minibatch_size, )}
# data_shapes = {'data': (minibatch_size, history_length) + game.replay_memory.state_dim,
#                'policy_reward': (minibatch_size, )}


normal_output_op = LogNormalPolicy()
policy_net_mean, policy_net_var, policy_net = normal_policy_sym(
    game.replay_memory.action_dim[0], normal_output_op)

net = Base(data_shapes=data_shapes, sym=policy_net, name='PolicyNet',
           initializer=mx.initializer.Xavier(factor_type='in', magnitude=1.0), ctx=ctx)
optimizer = mx.optimizer.create(name='sgd', learning_rate=0.00001,
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
        game.begin_episode(steps_left)
        time_episode_start = time.time()
        while not game.episode_terminate():
            # choose action
            if game.state_enabled and game.replay_memory.sample_enabled:
                do_exploration = (npy_rng.rand() < eps_curr)
                eps_curr = max(eps_curr - eps_decay, eps_min)
                if do_exploration:
                    action = npy_rng.normal(5.335, 0.1, action_dim)
                else:
                    current_state = game.current_state()
                    # note that current_state shape is history_length by state_dim
                    # for now since history length = 1 we only take the 1 step state
                    # state normalization?
                    if history_length == 1:
                        state = current_state
                    else:
                        state = current_state.reshape((1, ) + current_state.shape)
                    policy_means = net.forward(sym_name="fc_mean_3_output", batch_size=1,
                                               is_train=False, data=state)[0].asnumpy()
                    policy_vars = net.forward(sym_name='fc_var_softplus_1_output',
                                              batch_size=1,
                                              is_train=False, data=state)[0].asnumpy()
                    action = net.forward(is_train=False, batch_size=1,
                                         data=state)[0].asnumpy()
                    action = action.reshape((action.shape[1], ))
            else:
                action = npy_rng.normal(5.335, 0.1, action_dim)
            # play the game
            game.play(action)
            total_steps += 1

            # update the policy network
            if total_steps % update_interval == 0 and game.replay_memory.sample_enabled:
                # draw sample from the replay memory
                training_steps += 1
                states, actions, rewards, terminate_flag = \
                    game.replay_memory.sample_trajectory(episode_length=episode_T)
                curbatch_size = states.shape[0]
                # how to choose baseline?
                rewards = numpy.cumsum(rewards[::-1], axis=0)[::-1]
                baseline -= 0.01 * (baseline - rewards.mean(axis=0))
                policy_means = net.forward(sym_name='fc_mean_3_output',
                                           batch_size=curbatch_size,
                                           is_train=False, data=states)[0].asnumpy()
                policy_vars = net.forward(sym_name='fc_var_softplus_1_output',
                                          batch_size=curbatch_size, is_train=False,
                                          data=states)[0].asnumpy()
                policy_actions = net.forward(is_train=True, batch_size=curbatch_size,
                                             data=states)[0].asnumpy()
                net.backward(batch_size=curbatch_size, policy_score=rewards - baseline)
                net.update(updater)

                # calculate loss
                loss = -numpy.log(numpy.sqrt(policy_vars)) + 0.5*numpy.square(policy_means-policy_actions) / policy_vars
                loss = numpy.sum(loss, axis=1) * (rewards - baseline)

        time_episode_end = time.time()
        steps_left -= game.episode_step
        # update the statistics
        epoch_reward += game.episode_reward
        info_str = 'Epoch:%d, Episode%d, Steps Left:%d/%d, Reward:%f, fps:%f, Exploration:%f' \
                   % (epoch, episode, steps_left, steps_per_epoch, game.episode_reward, game.episode_step / (time_episode_end - time_episode_start), eps_curr)
        print info_str

    end_epoch = time.time()
    net.save_params('./models/', epoch=epoch)
    print 'Epoch:%d, FPS:%f, Avg Reward: %f/%d' \
          % (epoch, steps_per_epoch / (end_epoch-start_epoch), epoch_reward / float(episode), episode)



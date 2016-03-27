import mxnet as mx
import mxnet.ndarray as nd
from mxnet import kvstore
import numpy
from arena import Base
from arena.games import AtariGame
from arena.utils import *
import logging
import argparse
import os
import re
import sys
import time

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
mx.random.seed(100)
npy_rng = get_numpy_rng()


# TODO NDArrayOP will cause some troubles see `https://github.com/dmlc/mxnet/issues/1720'
class DQNOutputOp(mx.operator.NDArrayOp):
    def __init__(self):
        super(DQNOutputOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        reward_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, reward_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = out_data[0]
        action = in_data[1]
        reward = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        dx[:] = nd.fill_element_0index(dx,
                                       nd.clip(nd.choose_element_0index(x, action) - reward, -1, 1),
                                       action)

def update_to_kvstore(kvStore,params,params_grad):
    for paramIndex in range(len(params)):
        k=params.keys()[paramIndex]
        kvStore.push(paramIndex,params_grad[k],priority=-paramIndex)
        timeAfterPush = time.time()
        kvStore.pull(paramIndex,params[k],priority=-paramIndex)
# TODO Regression Output has none differential for label, we may need to fix that
class DQNOutputNpyOp(mx.operator.NumpyOp):
    def __init__(self):
        super(DQNOutputNpyOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        reward_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, reward_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = out_data[0]
        action = in_data[1].astype(numpy.int)
        reward = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        dx[numpy.arange(action.shape[0]), action] \
            = numpy.clip(x[numpy.arange(action.shape[0]), action] - reward, -1, 1)


def dqn_sym_nips(action_num, output_op):
    net = mx.symbol.Variable('data')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=action_num)
    net = output_op(data=net, name='dqn')
    return net


def dqn_sym_nature(action_num, output_op):
    net = mx.symbol.Variable('data')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=64)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv3', kernel=(3, 3), stride=(1, 1), num_filter=64)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=512)
    net = mx.symbol.Activation(data=net, name='relu4', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc5', num_hidden=action_num)
    net = output_op(data=net, name='dqn')
    return net


class DQNInitializer(mx.initializer.Xavier):
    def _init_bias(self, _, arr):
        arr[:] = .1


def main():
    parser = argparse.ArgumentParser(description='Script to test the trained network on a game.')
    parser.add_argument('-r', '--rom', required=False, type=str,
                        default=os.path.join('arena', 'games', 'roms', 'breakout.bin'),
                        help='Path of the ROM File.')
    parser.add_argument('-v', '--visualization', required=False, type=int, default=0,
                        help='Visualize the runs.')
    parser.add_argument('--lr', required=False, type=float, default=0.01,
                        help='Learning rate of the AdaGrad optimizer')
    parser.add_argument('--eps', required=False, type=float, default=0.01,
                        help='Eps of the AdaGrad optimizer')
    parser.add_argument('--clip-gradient', required=False, type=float, default=None,
                        help='Clip threshold of the AdaGrad optimizer')
    parser.add_argument('--double-q', required=False, type=bool, default=False,
                        help='Use Double DQN')
    parser.add_argument('--wd', required=False, type=float, default=0.0,
                        help='Weight of the L2 Regularizer')
    parser.add_argument('-c', '--ctx', required=False, type=str, default='gpu',
                        help='Running Context. E.g `-c gpu` or `-c gpu1` or `-c cpu`')
    parser.add_argument('-d', '--dir-path', required=False, type=str, default='',
                        help='Saving directory of model files.')
    parser.add_argument('--start-eps', required=False, type=float, default=1.0,
                        help='Eps of the epsilon-greedy policy at the beginning')
    parser.add_argument('--replay-start-size', required=False, type=int, default=50000,
                        help='The step that the training starts')
    parser.add_argument('--kvstore-update-period', required=False, type=int, default=1,
                        help='The period that the worker updates the parameters from the sever')
    parser.add_argument('--kv-type', required=False, type=str, default=None,
                        help='type of kvstore, default will not use kvstore, could also be dist_async')
    args, unknown = parser.parse_known_args()
    if args.dir_path == '':
        rom_name = os.path.splitext(os.path.basename(args.rom))[0]
        args.dir_path = 'dqn-%s' % rom_name
    ctx = re.findall('([a-z]+)(\d*)', args.ctx)
    ctx = [(device, int(num)) if len(num) >0 else (device, 0) for device, num in ctx]
    replay_start_size = args.replay_start_size
    max_start_nullops = 30
    replay_memory_size = 1000000
    history_length = 4
    rows = 84
    cols = 84
    q_ctx = mx.Context(*ctx[0])

    game = AtariGame(rom_path=args.rom, resize_mode='scale', replay_start_size=replay_start_size,
                     resized_rows=rows, resized_cols=cols, max_null_op=max_start_nullops,
                     replay_memory_size=replay_memory_size, display_screen=args.visualization,
                     history_length=history_length)

    ##RUN NATURE
    freeze_interval = 10000
    epoch_num = 200
    steps_per_epoch = 250000
    update_interval = 4
    discount = 0.99

    eps_start = args.start_eps
    eps_min = 0.1
    eps_decay = (eps_start - 0.1) / 1000000
    eps_curr = eps_start
    freeze_interval /= update_interval
    minibatch_size = 32
    action_num = len(game.action_set)

    data_shapes = {'data': (minibatch_size, history_length) + (rows, cols),
                   'dqn_action': (minibatch_size,), 'dqn_reward': (minibatch_size,)}
    
    optimizer = mx.optimizer.create(name='adagrad', learning_rate=args.lr, eps=args.eps,
                        clip_gradient=args.clip_gradient,
                        rescale_grad=1.0, wd=args.wd)
    dqn_output_op = DQNOutputNpyOp()
    dqn_sym = dqn_sym_nature(action_num, dqn_output_op)
    qnet = Base(data_shapes=data_shapes, sym=dqn_sym, name='QNet',
                  initializer=DQNInitializer(factor_type="in"),
                  ctx=q_ctx)
    target_qnet = qnet.copy(name="TargetQNet", ctx=q_ctx)
    # Create kvstore
    if args.kv_type != None:
        kvType = args.kv_type
        kvStore = kvstore.create(kvType)
        #Initialize kvstore
        for idx,v in enumerate(qnet.params.values()):
            kvStore.init(idx,v);
        # Set optimizer on kvstore
        kvStore.set_optimizer(optimizer)
        kvstore_update_period = args.kvstore_update_period
    else:
        updater = mx.optimizer.get_updater(optimizer)

    qnet.print_stat()
    target_qnet.print_stat()
    # Begin Playing Game
    training_steps = 0
    total_steps = 0
    for epoch in xrange(epoch_num):
        # Run Epoch
        steps_left = steps_per_epoch
        episode = 0
        epoch_reward = 0
        start = time.time()
        game.start()
        while steps_left > 0:
            # Running New Episode
            episode += 1
            episode_loss = 0.0
            episode_q_value = 0.0
            episode_update_step = 0
            episode_action_step = 0
            time_episode_start = time.time()
            game.begin_episode(steps_left)
            while not game.episode_terminate:
                # 1. We need to choose a new action based on the current game status
                if game.state_enabled and game.replay_memory.sample_enabled:
                    do_exploration = (npy_rng.rand() < eps_curr)
                    eps_curr = max(eps_curr - eps_decay, eps_min)
                    if do_exploration:
                        action = npy_rng.randint(action_num)
                    else:
                        # TODO Here we can in fact play multiple gaming instances simultaneously and make actions for each
                        # We can simply stack the current_state() of gaming instances and give prediction for all of them
                        # We need to wait after calling calc_score(.), which makes the program slow
                        # TODO Profiling the speed of this part!
                        current_state = game.current_state()
                        state = nd.array(current_state.reshape((1,) + current_state.shape),
                                         ctx=q_ctx) / float(255.0)
                        qval_npy = qnet.forward(batch_size=1, data=state)[0].asnumpy()
                        action = numpy.argmax(qval_npy)
                        episode_q_value += qval_npy[0, action]
                        episode_action_step += 1
                else:
                    action = npy_rng.randint(action_num)

                # 2. Play the game for a single mega-step (Inside the game, the action may be repeated for several times)
                game.play(action)
                total_steps += 1

                # 3. Update our Q network if we can start sampling from the replay memory
                #    Also, we update every `update_interval`
                if total_steps % update_interval == 0 and game.replay_memory.sample_enabled:
                    # 3.1 Draw sample from the replay_memory
                    training_steps += 1
                    episode_update_step += 1
                    states, actions, rewards, next_states, terminate_flags \
                        = game.replay_memory.sample(batch_size=minibatch_size)
                    states = nd.array(states, ctx=q_ctx) / float(255.0)
                    next_states = nd.array(next_states, ctx=q_ctx) / float(255.0)
                    actions = nd.array(actions, ctx=q_ctx)
                    rewards = nd.array(rewards, ctx=q_ctx)
                    terminate_flags = nd.array(terminate_flags, ctx=q_ctx)

                    # 3.2 Use the target network to compute the scores and
                    #     get the corresponding target rewards
                    if not args.double_q:
                        target_qval = target_qnet.forward(batch_size=minibatch_size,
                                                         data=next_states)[0]
                        target_rewards = rewards + nd.choose_element_0index(target_qval,
                                                                nd.argmax_channel(target_qval))\
                                           * (1.0 - terminate_flags) * discount
                    else:
                        target_qval = target_qnet.forward(batch_size=minibatch_size,
                                                         data=next_states)[0]
                        qval = qnet.forward(batch_size=minibatch_size, data=next_states)[0]

                        target_rewards = rewards + nd.choose_element_0index(target_qval,
                                                                nd.argmax_channel(qval))\
                                           * (1.0 - terminate_flags) * discount
                    outputs = qnet.forward(batch_size=minibatch_size,is_train=True, data=states,
                                              dqn_action=actions,
                                              dqn_reward=target_rewards)
                    qnet.backward(batch_size=minibatch_size)
                    # time_before_update = time.time()
                    if args.kv_type != None:
                        if total_steps % kvstore_update_period == 0:
                            update_to_kvstore(kvStore,qnet.params,qnet.params_grad)
                    else:
                        qnet.update(updater=updater)

                    # logging.info("update time %f" %(time.time()-time_before_update))
                    # time_before_wait = time.time()
                    # for v in qnet.params.values():
                    #     v.wait_to_read()
                    # logging.info("wait time %f" %(time.time()-time_before_wait))

                    # 3.3 Calculate Loss
                    diff = nd.abs(nd.choose_element_0index(outputs[0], actions) - target_rewards)
                    quadratic_part = nd.clip(diff, -1, 1)
                    loss = (0.5 * nd.sum(nd.square(quadratic_part)) + nd.sum(diff - quadratic_part)).asscalar()
                    episode_loss += loss

                    # 3.3 Update the target network every freeze_interval
                    # (We can do annealing instead of hard copy)
                    if training_steps % freeze_interval == 0:
                        qnet.copy_params_to(target_qnet)
            steps_left -= game.episode_step
            time_episode_end = time.time()
            # Update the statistics
            epoch_reward += game.episode_reward
            info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d, Reward:%f, fps:%f, Exploration:%f" \
                        % (epoch, episode, steps_left, steps_per_epoch, game.episode_reward,
                           game.episode_step / (time_episode_end - time_episode_start), eps_curr)
            if episode_update_step > 0:
                info_str += ", Avg Loss:%f/%d" % (episode_loss / episode_update_step,
                                                  episode_update_step)
            if episode_action_step > 0:
                info_str += ", Avg Q Value:%f/%d" % (episode_q_value / episode_action_step,
                                                  episode_action_step)
            logging.info(info_str)
        end = time.time()
        fps = steps_per_epoch / (end - start)
        qnet.save_params(dir_path=args.dir_path, epoch=epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, epoch_reward / float(episode), episode))

if __name__ == '__main__':
    main()

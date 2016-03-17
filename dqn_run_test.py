import mxnet as mx
import mxnet.ndarray as nd
import numpy
import time
import argparse
import os
import re
import logging
import sys
from arena import Critic
from arena.games import AtariGame


root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


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


def collect_holdout_samples(game, num_steps=3200, sample_num=3200):
    print "Begin Collecting Holdout Samples...",
    game.force_restart()
    game.begin_episode()
    for i in xrange(num_steps):
        if game.episode_terminate:
            game.begin_episode()
        action = numpy.random.randint(len(game.action_set))
        game.play(action)
    samples, _, _, _, _ = game.replay_memory.sample(batch_size=sample_num)
    print "Done!"
    return samples


def calculate_avg_q(samples, qnet):
    total_q = 0.0
    for i in xrange(len(samples)):
        state = nd.array(samples[i:i + 1], ctx=qnet.ctx) / float(255.0)
        total_q += qnet.calc_score(batch_size=1, data=state)[0].asnumpy().max(axis=1).sum()
    avg_q_score = total_q / float(len(samples))
    return avg_q_score


def calculate_avg_reward(game, qnet, test_steps=125000, exploartion=0.05):
    game.force_restart()
    action_num = len(game.action_set)
    total_reward = 0
    steps_left = test_steps
    episode = 0
    while steps_left > 0:
        # Running New Episode
        episode += 1
        episode_q_value = 0.0
        game.begin_episode(steps_left)
        start = time.time()
        while not game.episode_terminate:
            # 1. We need to choose a new action based on the current game status
            if game.state_enabled:
                do_exploration = (numpy.random.rand() < exploartion)
                if do_exploration:
                    action = numpy.random.randint(action_num)
                else:
                    # TODO Here we can in fact play multiple gaming instances simultaneously and make actions for each
                    # We can simply stack the current_state() of gaming instances and give prediction for all of them
                    # We need to wait after calling calc_score(.), which makes the program slow
                    # TODO Profiling the speed of this part!
                    current_state = game.current_state()
                    state = nd.array(current_state.reshape((1,) + current_state.shape),
                                     ctx=qnet.ctx) / float(255.0)
                    action = nd.argmax_channel(
                        qnet.calc_score(batch_size=1, data=state)[0]).asscalar()
            else:
                action = numpy.random.randint(action_num)

            # 2. Play the game for a single mega-step (Inside the game, the action may be repeated for several times)
            game.play(action)
        end = time.time()
        steps_left -= game.episode_step
        print 'Episode:%d, FPS:%s, Steps Left:%d, Reward:%d' \
              % (episode, game.episode_step / (end - start), steps_left, game.episode_reward)
        total_reward += game.episode_reward
    avg_reward = total_reward / float(episode)
    return avg_reward


def main():
    parser = argparse.ArgumentParser(description='Script to test the trained network on a game.')
    parser.add_argument('-r', '--rom', required=False, type=str,
                        default=os.path.join('arena', 'games', 'roms', 'breakout.bin'),
                        help='Path of the ROM File.')
    parser.add_argument('-d', '--dir-path', required=False, type=str, default='',
                        help='Directory path of the model files.')
    parser.add_argument('-m', '--model_prefix', required=True, type=str, default='QNet',
                        help='Prefix of the saved model file.')
    parser.add_argument('-t', '--test-steps', required=False, type=int, default=125000,
                        help='Test steps.')
    parser.add_argument('-c', '--ctx', required=False, type=str, default='gpu',
                        help='Running Context. E.g `-c gpu` or `-c gpu1` or `-c cpu`')
    parser.add_argument('-e', '--epoch-range', required=False, type=str, default='22',
                        help='Epochs to run testing. E.g `-e 0,80`, `-e 0,80,2`')
    parser.add_argument('-v', '--visualization', required=False, type=int, default=0,
                        help='Visualize the runs.')
    args, unknown = parser.parse_known_args()
    max_start_nullops = 30
    holdout_size = 3200
    replay_memory_size = 1000000
    exploartion = 0.05
    history_length = 4
    rows = 84
    cols = 84
    ctx = re.findall('([a-z]+)(\d*)', args.ctx)
    ctx = [(device, int(num)) if len(num) >0 else (device, 0) for device, num in ctx]
    q_ctx = mx.Context(*ctx[0])
    minibatch_size = 32
    epoch_range = [int(n) for n in args.epoch_range.split(',')]
    epochs = range(*epoch_range)

    game = AtariGame(rom_path=args.rom, history_length=history_length,
                     resize_mode='scale', resized_rows=rows, replay_start_size=4,
                     resized_cols=cols, max_null_op=max_start_nullops,
                     replay_memory_size=replay_memory_size,
                     death_end_episode=False,
                     display_screen=args.visualization)
    ch = raw_input()
    if not args.visualization:
        holdout_samples = collect_holdout_samples(game, sample_num=holdout_size)
    action_num = len(game.action_set)
    data_shapes = {'data': (minibatch_size, history_length) + (rows, cols),
                   'dqn_action': (minibatch_size,), 'dqn_reward': (minibatch_size,)}
    dqn_output_op = DQNOutputOp()
    dqn_sym = dqn_sym_nature(action_num, dqn_output_op)
    qnet = Critic(data_shapes=data_shapes, sym=dqn_sym, name=args.model_prefix, ctx=q_ctx)

    for epoch in epochs:
        qnet.load_params(name=args.model_prefix, dir_path=args.dir_path, epoch=epoch)
        if not args.visualization:
            avg_q_score = calculate_avg_q(holdout_samples, qnet)
            avg_reward = calculate_avg_reward(game, qnet, args.test_steps, exploartion)
            print "Epoch:%d Avg Reward: %f, Avg Q Score:%f" % (epoch, avg_reward, avg_q_score)
        else:
            avg_reward = calculate_avg_reward(game, qnet, args.test_steps, exploartion)
            print "Epoch:%d Avg Reward: %f" % (epoch, avg_reward)

if __name__ == '__main__':
    main()
import mxnet as mx
import mxnet.ndarray as nd
import numpy
import time
from arena import Critic
from arena.games import AtariGame

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

max_start_nullops = 30
replay_memory_size = 1000000
rows = 84
cols = 84
q_ctx = mx.gpu()
minibatch_size = 32

game = AtariGame(resize_mode='scale', resized_rows=rows,
                 resized_cols=cols, max_null_op=max_start_nullops,
                 replay_memory_size=replay_memory_size,
                 death_end_episode=False,
                 display_screen=False)
action_num = len(game.action_set)

data_shapes = {'data': (minibatch_size, action_num) + (rows, cols),
               'dqn_action': (minibatch_size,), 'dqn_reward': (minibatch_size,)}
dqn_output_op = DQNOutputOp()
dqn_sym = dqn_sym_nature(action_num, dqn_output_op)
qnet = Critic(data_shapes=data_shapes, sym=dqn_sym, name='QNet', ctx=q_ctx)
qnet.load_params(name='QNet', dir_path='dqn-model', epoch=51)

test_steps = 125000
eps_curr = 0.05
episode = 0
total_reward = 0

steps_left = test_steps
while steps_left > 0:
    # Running New Episode
    episode += 1
    episode_q_value = 0.0
    game.begin_episode(steps_left)
    start = time.time()
    while not game.episode_terminate:
        # 1. We need to choose a new action based on the current game status
        if game.state_enabled:
            do_exploration = (numpy.random.rand() < eps_curr)
            if do_exploration:
                action = numpy.random.randint(action_num)
            else:
                # TODO Here we can in fact play multiple gaming instances simultaneously and make actions for each
                # We can simply stack the current_state() of gaming instances and give prediction for all of them
                # We need to wait after calling calc_score(.), which makes the program slow
                # TODO Profiling the speed of this part!
                current_state = game.current_state()
                state = nd.array(current_state.reshape((1,) + current_state.shape),
                                 ctx=q_ctx) / float(255.0)
                action = nd.argmax_channel(
                    qnet.calc_score(batch_size=1, data=state)[0]).asscalar()
                # action = q_score.argmax(axis=1)[0]
                # episode_q_value += q_score[0, action]
                # episode_act_step += 1
        else:
            action = numpy.random.randint(action_num)

        # 2. Play the game for a single mega-step (Inside the game, the action may be repeated for several times)
        game.play(action)
    end = time.time()
    steps_left -= game.episode_step
    print 'Episode:%d, FPS:%s, Steps Left:%d, Reward:%d' \
          %(episode, game.episode_step/(end-start), steps_left, game.episode_reward)
    total_reward += game.episode_reward
avg_reward = total_reward/float(episode)

total_q = 0.0
for i in xrange(100):
    state, _, _, _, _ = game.replay_memory.sample(batch_size=32)
    state = nd.array(state, ctx=q_ctx) / float(255.0)
    total_q += qnet.calc_score(batch_size=32, data=state)[0].asnumpy().max(axis=1).sum()
avg_q_score = total_q / float(3200)
print "Avg Reward: %f, Avg Q Score:%f" %(avg_reward, avg_q_score)
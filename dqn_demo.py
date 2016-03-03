import mxnet as mx
import mxnet.ndarray as nd
import numpy
from arena import Critic
from arena.games import AtariGame
import logging
import sys
import time

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


# TODO Regression Output has none differential for label, we may need to fix that
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
        action = in_data[1]  # .astype('int')
        reward = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        dx[:] = nd.fill_element_0index(dx,
        #                               nd.clip(nd.choose_element_0index(x, action) - reward, -1, 1),
                                       nd.choose_element_0index(x, action) - reward,
                                       action)


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


class DQNInitializer(mx.initializer.Normal):
    def _init_bias(self, _, arr):
        arr[:] = .1


replay_start_size = 1000
max_start_nullops = 30
replay_memory_size = 1000000
rows = 84
cols = 84
q_ctx = mx.gpu()
target_q_ctx = mx.gpu()

game = AtariGame(resize_mode='crop', replay_start_size=replay_start_size, resized_rows=rows,
                 resized_cols=cols, max_null_op=max_start_nullops,
                 replay_memory_size=replay_memory_size, display_screen=False)

'''
##RUN NATURE
freeze_interval = 10000
epoch_num = 200
steps_per_epoch = 250000
steps_per_test = 125000
update_interval = 4
discount = 0.99
'''

##RUN NATURE
freeze_interval = 1
epoch_num = 100
steps_per_epoch = 50000
steps_per_test = 10000
update_interval = 1
discount = 0.95

eps_start = 1.0
eps_min = 0.1
eps_decay = (1.0 - 0.1) / 1000000
eps_curr = eps_start
freeze_interval /= update_interval
minibatch_size = 32
action_num = len(game.action_set)

data_shapes = {'data': (minibatch_size, action_num) + (rows, cols),
               'dqn_action': (minibatch_size,), 'dqn_reward': (minibatch_size,)}
optimizer_params = {'name': 'adam', 'learning_rate': 0.0002,
                    'rescale_grad': 1.0 / float(minibatch_size),
                    'wd': 0}
dqn_output_op = DQNOutputOp()
dqn_sym = dqn_sym_nips(action_num, dqn_output_op)
qnet = Critic(data_shapes=data_shapes, sym=dqn_sym, optimizer_params=optimizer_params, name='QNet',
              initializer=DQNInitializer(),
              ctx=q_ctx)
target_qnet = qnet.copyto("TargetQNet", ctx=target_q_ctx)

# Begin Playing Game
training_steps = 0
game.start()
for epoch in xrange(epoch_num):
    # Run Epoch
    steps_left = steps_per_epoch
    episode = 0
    epoch_reward = 0
    epoch_q_value = 0
    epoch_act_step = 0
    start = time.time()
    while steps_left > 0:
        # Running New Episode
        episode += 1
        episode_loss = 0.0
        episode_update_step = 0
        episode_q_value = 0.0
        episode_act_step = 0
        game.begin_episode(steps_left)
        while not game.episode_terminate:
            # 1. We need to choose a new action based on the current game status
            if game.state_enabled and game.replay_memory.sample_enabled:
                do_exploration = (numpy.random.rand() < eps_curr)
                eps_curr = max(eps_curr - eps_decay, eps_min)
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
                    q_score = qnet.calc_score(batch_size=1, data=state)[0].asnumpy()
                    action = q_score.argmax(axis=1)[0]
                    episode_q_value += q_score[0, action]
                    episode_act_step += 1
            else:
                action = numpy.random.randint(action_num)

            # 2. Play the game for a single mega-step (Inside the game, the action may be repeated for several times)
            game.play(action)

            # 3. Update our Q network if we can start sampling from the replay memory
            #    Also, we update every `update_interval`
            if game.episode_step % update_interval == 0 and game.replay_memory.sample_enabled:
                # 3.1 Draw sample from the replay_memory
                training_steps += 1
                episode_update_step += 1
                states, actions, rewards, next_states, terminate_flags \
                    = game.replay_memory.sample(batch_size=minibatch_size)
                states = nd.array(states, ctx=q_ctx) / float(255.0)
                next_states = nd.array(next_states, ctx=target_q_ctx) / float(255.0)

                # 3.2 Use the target network to compute the scores and get the corresponding target rewards

                target_qval = target_qnet.calc_score(batch_size=minibatch_size,
                                                     data=next_states)[0].asnumpy()
                ind = (1 - terminate_flags.ravel()).nonzero()[0]
                if len(ind) > 0:
                    rewards[ind] += discount * numpy.max(target_qval[ind, ...], axis=1)
                outputs = qnet.fit_target(batch_size=minibatch_size, data=states, dqn_action=actions,
                                          dqn_reward=rewards)
                #print "Chosed:", nd.choose_element_0index(outputs[0], nd.array(actions, ctx=outputs[0].context)).asnumpy()
                #print "Rewards:", rewards
                loss = numpy.sqrt(0.5*numpy.square(nd.choose_element_0index(outputs[0],
                            nd.array(actions, ctx=outputs[0].context)).asnumpy() - rewards).mean())
                episode_loss += loss

                # 3.3 Update the target network every freeze_interval
                # (We can do annealing instead of hard copy)
                if training_steps % freeze_interval == 0:
                    qnet.copy_params_to(target_qnet)
        steps_left -= game.episode_step

        # Update the statistics
        epoch_reward += game.episode_reward
        logging.info("Episode:%d, Steps Left:%d/%d, Reward:%f, Exploration:%f"
                     % (episode, steps_left, steps_per_epoch, game.episode_reward,
                        eps_curr))
        if episode_update_step > 0:
            logging.info("Avg Loss:%f/%d" % (episode_loss / episode_update_step, episode_update_step))
        if episode_act_step > 0:
            logging.info("Avg Q Value:%f/%d" % (episode_q_value / episode_act_step, episode_act_step))
    end = time.time()
    fps = steps_per_epoch / (end - start)
    logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d")
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
from collections import OrderedDict
from arena.operators import *

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
mx.random.seed(100)
npy_rng = get_numpy_rng()


class EpisodeStat(object):
    def __init__(self):
        self.episode_loss = 0.0
        self.episode_q_value = 0.0
        self.episode_update_step = 0
        self.episode_action_step = 0


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
    parser.add_argument('--optimizer', required=False, type=str, default="adagrad",
                        help='type of optimizer')
    args, unknown = parser.parse_known_args()
    if args.dir_path == '':
        rom_name = os.path.splitext(os.path.basename(args.rom))[0]
        args.dir_path = 'dqn-%s-%de_5' % (rom_name,int(args.lr*10**5))
    ctx = re.findall('([a-z]+)(\d*)', args.ctx)
    ctx = [(device, int(num)) if len(num) >0 else (device, 0) for device, num in ctx]

    # Async verision
    nactor= 16
    param_update_period = 5

    replay_start_size = args.replay_start_size
    max_start_nullops = 30
    replay_memory_size = 500
    history_length = 4
    rows = 84
    cols = 84
    q_ctx = mx.Context(*ctx[0])
    games = []
    # TODO:Build a list of games
    for g in range(nactor):
        games.append(AtariGame(rom_path=args.rom, resize_mode='scale', replay_start_size=replay_start_size,
                             resized_rows=rows, resized_cols=cols, max_null_op=max_start_nullops,
                             replay_memory_size=replay_memory_size, display_screen=args.visualization,
                             history_length=history_length))


    ##RUN NATURE
    freeze_interval = 40000
    epoch_num = 200
    steps_per_epoch = 250000
    update_interval = 4
    discount = 0.99

    eps_start = args.start_eps
    eps_min = 0.1
    eps_decay = (eps_start - 0.1) / 1000000
    eps_curr = eps_start
    freeze_interval /= update_interval
    minibatch_size = nactor * param_update_period
    action_num = len(games[0].action_set)

    data_shapes = {'data': (minibatch_size, history_length) + (rows, cols),
                   'dqn_action': (minibatch_size,), 'dqn_reward': (minibatch_size,)}

    dqn_output_op = DQNOutputNpyOp()
    dqn_sym = dqn_sym_nips(action_num, dqn_output_op)
    qnet = Base(data_shapes=data_shapes, sym=dqn_sym, name='QNet',
                  initializer=DQNInitializer(factor_type="in"),
                  ctx=q_ctx)
    target_qnet = qnet.copy(name="TargetQNet", ctx=q_ctx)

    use_easgd = False
    if args.optimizer != "easgd":
        optimizer = mx.optimizer.create(name=args.optimizer, learning_rate=args.lr, eps=args.eps,
                        clip_gradient=args.clip_gradient,
                        rescale_grad=1.0, wd=args.wd)
    else:
        use_easgd = True
        easgd_beta = 0.9
        easgd_p = 4
        easgd_alpha = easgd_beta/(args.kvstore_update_period*easgd_p)
        optimizer = mx.optimizer.create(name="ServerEasgd",learning_rate=easgd_alpha)
        easgd_eta = 0.00025
        local_optimizer = mx.optimizer.create(name='adagrad', learning_rate=args.lr, eps=args.eps,
                        clip_gradient=args.clip_gradient,
                        rescale_grad=1.0, wd=args.wd)
        central_weight = OrderedDict([(n, nd.zeros(v.shape, ctx=q_ctx))
                                        for n, v in qnet.params.items()])
    # Create kvstore
    if args.kv_type != None:
        kvType = args.kv_type
        kv = kvstore.create(kvType)
        #Initialize kvstore
        for idx,v in enumerate(qnet.params.values()):
            kv.init(idx,v);
        if use_easgd == False:
            # Set optimizer on kvstore
            kv.set_optimizer(optimizer)
        else:
            # kv.send_updater_to_server(easgd_server_update)
            kv.set_optimizer(optimizer)
            local_updater = mx.optimizer.get_updater(local_optimizer)
        kvstore_update_period = args.kvstore_update_period
        args.dir_path = args.dir_path + "-"+str(kv.rank)
    else:
        updater = mx.optimizer.get_updater(optimizer)

    qnet.print_stat()
    target_qnet.print_stat()
    # Begin Playing Game
    training_steps = 0
    total_steps = 0
    ave_fps = 0
    ave_loss = 0
    time_for_info = time.time()
    for epoch in xrange(epoch_num):
        # Run Epoch
        steps_left = steps_per_epoch
        episode = 0
        epoch_reward = 0
        start = time.time()
        #
        for game in games:
            game.start()
            game.begin_episode()
        episode_stats = [EpisodeStat() for i in range(len(games))]
        while steps_left > 0:

            for g, game in enumerate(games):
                if game.episode_terminate:
                    episode += 1
                    epoch_reward += game.episode_reward
                    if args.kv_type != None:
                        info_str="Node[%d]: " %kv.rank
                    else:
                        info_str =""
                    info_str += "Epoch:%d, Episode:%d, Steps Left:%d/%d, Reward:%f, fps:%f, Exploration:%f" \
                                % (epoch, episode, steps_left, steps_per_epoch, game.episode_reward,
                                   ave_fps, eps_curr)
                    info_str += ", Avg Loss:%f" % ave_loss
                    if episode_stats[g].episode_action_step > 0:
                        info_str += ", Avg Q Value:%f/%d" % (episode_stats[g].episode_q_value / episode_stats[g].episode_action_step,
                                                          episode_stats[g].episode_action_step)
                    logging.info(info_str)

                    game.begin_episode(steps_left)
                    episode_stats[g] = EpisodeStat()
            for g, game in enumerate(games):
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
                        episode_stats[g].episode_q_value += qval_npy[0, action]
                        episode_stats[g].episode_action_step += 1
                else:
                    action = npy_rng.randint(action_num)

                # 2. Play the game for a single mega-step (Inside the game, the action may be repeated for several times)
                game.play(action)
                total_steps += 1
                if total_steps % 1000 == 0:
                    this_time = time.time()
                    ave_fps = (1000/(this_time-time_for_info))
                    time_for_info = this_time

                # 3. Update our Q network if we can start sampling from the replay memory
                #    Also, we update every `update_interval`
            if total_steps > replay_start_size*nactor and \
                total_steps % (nactor*param_update_period) == 0 and \
                games[-1].replay_memory.sample_enabled:
                # 3.1 Draw sample from the replay_memory
                training_steps += nactor

                for g,game in enumerate(games):
                    episode_stats[g].episode_update_step += 1
                    if g == 0:
                        states, actions, rewards, next_states, terminate_flags \
                            = game.replay_memory.sample(batch_size=minibatch_size/nactor)
                    else:
                        nstates, nactions, nrewards, nnext_states, nterminate_flags \
                            = game.replay_memory.sample(batch_size=minibatch_size/nactor)
                        states = numpy.concatenate((states,nstates))
                        actions = numpy.concatenate((actions,nactions))
                        rewards = numpy.concatenate((rewards,nrewards))
                        next_states = numpy.concatenate((next_states,nnext_states))
                        terminate_flags = numpy.concatenate((terminate_flags,nterminate_flags))
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


                if args.kv_type != None:
                    if total_steps % kvstore_update_period == 0:
                        if use_easgd == False:
                            update_to_kvstore(kv,qnet.params,qnet.params_grad)
                        else:
                            for paramIndex in range(len(qnet.params)):
                                k=qnet.params.keys()[paramIndex]
                                kv.pull(paramIndex,central_weight[k],priority=-paramIndex)
                                qnet.params[k][:] -= easgd_alpha*(qnet.params[k]-central_weight[k])
                                kv.push(paramIndex,qnet.params[k],priority=-paramIndex)
                    if use_easgd:
                        for paramIndex in range(len(qnet.params)):
                            k=qnet.params.keys()[paramIndex]
                            '''qnet.params[k][:] += -easgd_eta*nd.clip(qnet.params_grad[k],
                                                            -args.clip_gradient,
                                                            args.clip_gradient)'''
                            local_updater(index = paramIndex,grad=qnet.params_grad[k],
                                            weight=qnet.params[k])
                else:
                    qnet.update(updater=updater)



                # 3.3 Calculate Loss
                diff = nd.abs(nd.choose_element_0index(outputs[0], actions) - target_rewards)
                quadratic_part = nd.clip(diff, -1, 1)
                loss = (0.5 * nd.sum(nd.square(quadratic_part)) + nd.sum(diff - quadratic_part)).asscalar()
                if ave_loss == 0:
                    ave_loss =  loss
                else:
                    ave_loss =  0.95*ave_loss + 0.05*loss

                # 3.3 Update the target network every freeze_interval
                # (We can do annealing instead of hard copy)
                if training_steps % freeze_interval == 0:
                    qnet.copy_params_to(target_qnet)

            steps_left -= nactor





        end = time.time()
        fps = steps_per_epoch / (end - start)
        qnet.save_params(dir_path=args.dir_path, epoch=epoch)
        if args.kv_type != None:
            logging.info("Node[%d]: Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (kv.rank, epoch, fps, epoch_reward / float(episode), episode))
        else:
            logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, epoch_reward / float(episode), episode))

if __name__ == '__main__':
    main()

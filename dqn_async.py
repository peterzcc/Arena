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
import concurrent.futures
from functools import partial
root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
mx.random.seed(100)
npy_rng = get_numpy_rng()
def play_game(args):
    game,action = args
    game.play(action)


class EpisodeStat(object):
    def __init__(self):
        self.episode_loss = 0.0
        self.episode_q_value = 0.0
        self.episode_update_step = 0
        self.episode_action_step = 0

def sample_training_data(game,episode_stat,g):
    global states_buffer_for_train
    global actions_buffer_for_train
    global rewards_buffer_for_train
    global terminate_flags_buffer_for_train
    global  minibatch_size
    global nactor
    episode_stat.episode_update_step += 1
    single_size = minibatch_size/nactor
    action, reward, terminate_flag \
        = game.replay_memory.sample_inplace(batch_size=single_size,\
        states=states_buffer_for_train,offset=(g*single_size))
    actions_buffer_for_train[(g*single_size):((g+1)*single_size)]= action
    rewards_buffer_for_train[(g*single_size):((g+1)*single_size)]= reward
    terminate_flags_buffer_for_train[(g*single_size):((g+1)*single_size)]=\
        terminate_flag
total_steps=0
steps_left = 0
ave_fps = 0
ave_loss = 0
optimizer = None
episode_stats = []
def main():
    global steps_left
    global ave_fps
    global ave_loss
    global optimizer
    global episode_stats
    global total_steps
    global episode
    global epoch_reward
    global training_steps
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
    parser.add_argument('--rms-decay', required=False, type=float, default=0.95,
                        help='Decay rate of the RMSProp')
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
    parser.add_argument('--nactor', required=False, type=int, default=16,
                        help='number of actor')
    parser.add_argument('--exploration-period', required=False, type=int, default=4000000,
                        help='length of annealing of epsilon greedy policy')
    parser.add_argument('--replay-memory-size', required=False, type=int, default=100,
                        help='size of replay memory')
    parser.add_argument('--single-batch-size', required=False, type=int, default=5,
                        help='batch size for every actor')
    parser.add_argument('--symbol', required=False, type=str, default="nature",
                        help='type of network, nature or nips')
    parser.add_argument('--sample-policy', required=False, type=str, default="recent",
                        help='minibatch sampling policy, recent or random')
    parser.add_argument('--epoch-num', required=False, type=int, default=200,
                        help='number of epochs')
    args, unknown = parser.parse_known_args()

    if args.dir_path == '':
        rom_name = os.path.splitext(os.path.basename(args.rom))[0]
        time_str = time.strftime("%m%d-%H%M", time.gmtime())
        args.dir_path = ('dqn-%s-%d_' % (rom_name,int(args.lr*10**5)))+time_str
        logging.info("saving to dir: "+args.dir_path)
    ctx = re.findall('([a-z]+)(\d*)', args.ctx)
    ctx = [(device, int(num)) if len(num) >0 else (device, 0) for device, num in ctx]

    # Async verision
    nactor= args.nactor
    param_update_period = 5

    replay_start_size = args.replay_start_size
    max_start_nullops = 30
    replay_memory_size = args.replay_memory_size
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
    freeze_interval = 40000/nactor
    freeze_interval /= param_update_period
    epoch_num = args.epoch_num
    steps_per_epoch = 4000000/nactor
    discount = 0.99

    eps_start = numpy.ones((3,))* args.start_eps
    eps_min = numpy.array([0.1,0.01,0.5])
    eps_decay = (eps_start - eps_min) / (args.exploration_period/nactor)
    eps_curr = eps_start
    eps_id = numpy.zeros((nactor,))
    eps_update_period = 8000


    single_batch_size = args.single_batch_size
    minibatch_size = nactor * single_batch_size
    action_num = len(games[0].action_set)
    data_shapes = {'data': (minibatch_size, history_length) + (rows, cols),
                   'dqn_action': (minibatch_size,), 'dqn_reward': (minibatch_size,)}

    dqn_output_op = DQNOutputNpyOp()
    if args.symbol == "nature":
        dqn_sym = dqn_sym_nature(action_num, dqn_output_op)
    elif args.symbol == "nips":
        dqn_sym = dqn_sym_nips(action_num, dqn_output_op)
    qnet = Base(data_shapes=data_shapes, sym=dqn_sym, name='QNet',
                  initializer=DQNInitializer(factor_type="in"),
                  ctx=q_ctx)
    target_qnet = qnet.copy(name="TargetQNet", ctx=q_ctx)

    use_easgd = False
    if args.optimizer != "easgd":
        if args.optimizer == "adagrad":
            optimizer = mx.optimizer.create(name=args.optimizer, learning_rate=args.lr, eps=args.eps,
                            clip_gradient=args.clip_gradient,
                            rescale_grad=1.0, wd=args.wd)
        elif args.optimizer == "rmsprop":
            optimizer = mx.optimizer.create(name=args.optimizer, learning_rate=args.lr, eps=args.eps,
                            clip_gradient=args.clip_gradient,gamma1=args.rms_decay,gamma2=0,
                            rescale_grad=1.0, wd=args.wd)
            lr_decay = (args.lr - 0)/(steps_per_epoch*epoch_num/param_update_period)

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


    parallel_executor = concurrent.futures.ThreadPoolExecutor(nactor)
    for epoch in xrange(epoch_num):
        # Run Epoch

        steps_left = steps_per_epoch
        episode = 0
        epoch_reward = 0
        start = time.time()
        episode_stats = [EpisodeStat() for i in range(len(games))]
        def run_epoch(game,g,qnet=None,target_qnet=None,args=None,use_easgd=None,
                        updater=None,eps_curr=None,freeze_interval=None,
                        param_update_period=None,single_batch_size=None,
                        lr_decay=None,history_length=None):
            global steps_left
            global ave_fps
            global ave_loss
            global optimizer
            global episode_stats
            global total_steps
            global episode
            global epoch_reward
            global training_steps
            states_buffer_for_act = numpy.zeros((1, history_length)+(rows, cols),dtype='uint8')
            states_buffer_for_train = numpy.zeros((single_batch_size, history_length+1)+(rows, cols),dtype='uint8')
            game.start()
            game.begin_episode()
            time_for_info = time.time()
            while steps_left > 0:
                if total_steps % eps_update_period == 0:
                    eps_rand = npy_rng.rand()
                    if eps_rand<0.4:
                        eps_id[g] = 0
                    elif eps_rand<0.7:
                        eps_id[g] = 1
                    else:
                        eps_id[g] = 2
                if game.episode_terminate:
                    episode += 1
                    epoch_reward += game.episode_reward
                    if args.kv_type != None:
                        info_str="Node[%d]: " %kv.rank
                    else:
                        info_str =""
                    info_str += "Epoch:%d, Episode:%d, Steps Left:%d/%d, Reward:%f, fps:%f, Exploration:%f" \
                                % (epoch, episode, steps_left, steps_per_epoch, game.episode_reward,
                                   ave_fps, (eps_curr[eps_id[g]]))
                    info_str += ", Avg Loss:%f" % ave_loss
                    if episode_stats[g].episode_action_step > 0:
                        info_str += ", Avg Q Value:%f/%d" % (episode_stats[g].episode_q_value / episode_stats[g].episode_action_step,
                                                          episode_stats[g].episode_action_step)
                    logging.info(info_str)

                    game.begin_episode(steps_left)
                    episode_stats[g] = EpisodeStat()

                if game.replay_memory.size > history_length:
                    current_state = game.current_state()
                    states = nd.array(current_state.reshape((1,) + current_state.shape),ctx=q_ctx) / float(255.0)

                    qval_npy = qnet.forward(batch_size=1, data=states)[0].asnumpy()
                    actions_that_max_q = numpy.argmax(qval_npy)
                # 1. We need to choose a new action based on the current game status
                if game.state_enabled and game.replay_memory.sample_enabled:
                    do_exploration = (npy_rng.rand() < eps_curr[eps_id[g]])
                    if do_exploration:
                        action = npy_rng.randint(action_num)
                    else:
                        # TODO Here we can in fact play multiple gaming instances simultaneously and make actions for each
                        # We can simply stack the current_state() of gaming instances and give prediction for all of them
                        # We need to wait after calling calc_score(.), which makes the program slow
                        # TODO Profiling the speed of this part!
                        action = actions_that_max_q
                        episode_stats[g].episode_q_value += qval_npy[g, action]
                        episode_stats[g].episode_action_step += 1
                else:
                    action = npy_rng.randint(action_num)
                # for game,action in zip(games,actions):
                game.play(action)
                eps_curr = numpy.maximum(eps_curr - eps_decay, eps_min)
                total_steps += 1
                steps_left -= 1
                if total_steps % 100 == 0:
                    this_time = time.time()
                    ave_fps = (100/(this_time-time_for_info))
                    time_for_info = this_time

                    # 3. Update our Q network if we can start sampling from the replay memory
                    #    Also, we update every `update_interval`
                if total_steps > single_batch_size and \
                    total_steps % (param_update_period) == 0 and \
                    game.replay_memory.sample_enabled:
                    # 3.1 Draw sample from the replay_memory
                    training_steps += 1

                    # parallel_executor.map(sample_training_data,games,episode_stats,list(range(nactor)))
                    episode_stats[g].episode_update_step += 1
                    nsample = single_batch_size
                    if args.sample_policy == "recent":
                        action, reward, terminate_flag=game.replay_memory.sample_last(batch_size=nsample,\
                            states=states_buffer_for_train,offset=0)
                    elif args.sample_policy == "random":
                        action, reward, terminate_flag=game.replay_memory.sample_inplace(batch_size=nsample,\
                            states=states_buffer_for_train,offset=0)
                    states = nd.array(states_buffer_for_train[:,:-1], ctx=q_ctx) / float(255.0)
                    next_states = nd.array(states_buffer_for_train[:,1:], ctx=q_ctx) / float(255.0)
                    actions = nd.array(action, ctx=q_ctx)
                    rewards = nd.array(reward, ctx=q_ctx)
                    terminate_flags = nd.array(terminate_flag, ctx=q_ctx)

                    # 3.2 Use the target network to compute the scores and
                    #     get the corresponding target rewards
                    if not args.double_q:
                        target_qval = target_qnet.forward(batch_size=single_batch_size,
                                                         data=next_states)[0]
                        target_rewards = rewards + nd.choose_element_0index(target_qval,
                                                                nd.argmax_channel(target_qval))\
                                           * (1.0 - terminate_flags) * discount
                    else:
                        target_qval = target_qnet.forward(batch_size=single_batch_size,
                                                         data=next_states)[0]
                        qval = qnet.forward(batch_size=single_batch_size, data=next_states)[0]

                        target_rewards = rewards + nd.choose_element_0index(target_qval,
                                                                nd.argmax_channel(qval))\
                                           * (1.0 - terminate_flags) * discount

                    outputs = qnet.forward(batch_size=single_batch_size,is_train=True, data=states,
                                              dqn_action=actions,
                                              dqn_reward=target_rewards)
                    qnet.backward(batch_size=single_batch_size)


                    if args.kv_type != None:
                        if training_steps % kvstore_update_period == 0:
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
                                local_updater(index = paramIndex,grad=qnet.params_grad[k],
                                                weight=qnet.params[k])
                    else:
                        qnet.update(updater=updater)
                    if args.optimizer == "rmsprop":
                        optimizer.lr -= lr_decay

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
            return 0
        run_game = partial(run_epoch,qnet=qnet,target_qnet=target_qnet,args=args,use_easgd=use_easgd,
                            updater = updater,eps_curr=eps_curr,freeze_interval=freeze_interval,
                            param_update_period=param_update_period,lr_decay=lr_decay,
                            single_batch_size=single_batch_size,history_length=history_length)
        for result in parallel_executor.map(run_game,games,[g for g in range(nactor)]):
            pass
        # for g,game in enumerate(games):
        #     run_game(game,g)





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

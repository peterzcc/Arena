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
from threading import Lock
from threading import Thread
import cv2
root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
mx.random.seed(100)
npy_rng = get_numpy_rng()


class DQNInitializer(mx.initializer.Xavier):
    def _init_bias(self, _, arr):
        arr[:] = .1


def play_game(args):
    game,action = args
    game.play(action)


class EasgdThread(Thread):
    def __init__(self,kv,central_weight,local_weight,easgd_alpha,weight_write_lock,update_period,t_update):
        Thread.__init__(self)
        self.kv = kv
        self.central_weight = central_weight
        self.local_weight = local_weight
        self.easgd_alpha = easgd_alpha
        self.weight_write_lock = weight_write_lock
        self.update_period = update_period
        self.t_update = t_update
    def run(self):
        global training_steps
        prev_t = 0
        time_before = time.time()
        average_step = 0
        while (True):
            for paramIndex in xrange(len(self.local_weight)):
                k=self.local_weight.keys()[paramIndex]
                self.kv.pull(paramIndex,self.central_weight[k],priority=-paramIndex)
            for paramIndex in xrange(len(self.local_weight)):
                k=self.local_weight.keys()[paramIndex]
                self.central_weight[k].wait_to_read()
            if training_steps == prev_t:
                alpha = 1
            else:
                alpha = self.easgd_alpha * self.t_update / (training_steps - prev_t)
            with self.weight_write_lock:
                for paramIndex in xrange(len(self.local_weight)):
                    k=self.local_weight.keys()[paramIndex]
                    self.local_weight[k][:] -= alpha*(self.local_weight[k]-self.central_weight[k])
                    self.kv.push(paramIndex,self.local_weight[k],priority=-paramIndex)
            average_step = average_step*0.99 + (training_steps-prev_t)*0.01
            prev_t=training_steps
            time_current = time.time()
            if time_current > time_before + 10:
                logging.info("average step: %f" % average_step)
                time_before = time_current
            time.sleep(self.update_period)
class EpisodeStat(object):
    def __init__(self):
        self.episode_loss = 0.0
        self.episode_q_value = 0.0
        self.episode_update_step = 0
        self.episode_action_step = 0

def main():
    global training_steps
    training_steps = 0
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
    parser.add_argument('-c', '--ctx', required=False, type=str, default=None,
                        help='Running Context. E.g `-c gpu` or `-c gpu1` or `-c cpu`')
    parser.add_argument('-d', '--dir-path', required=False, type=str, default='',
                        help='Saving directory of model files.')
    parser.add_argument('--start-eps', required=False, type=float, default=1.0,
                        help='Eps of the epsilon-greedy policy at the beginning')
    parser.add_argument('--replay-start-size', required=False, type=int, default=50000,
                        help='The step that the training starts')
    parser.add_argument('--kvstore-update-period', required=False, type=int, default=16,
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
    parser.add_argument('--epoch-num', required=False, type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--param-update-period', required=False, type=int, default=5,
                        help='Parameter update period')
    parser.add_argument('--resize-mode', required=False, type=str, default="scale",
                        help='Resize mode, scale or crop')
    parser.add_argument('--eps-update-period', required=False, type=int, default=8000,
                        help='eps greedy policy update period')
    parser.add_argument('--server-optimizer', required=False, type=str, default="easgd",
                        help='type of server optimizer')
    parser.add_argument('--nworker', required=False, type=int, default=1,
                        help='number of kv worker')
    parser.add_argument('--easgd-update-period', required=False, type=float, default=0.3,
                        help='Update period of easgd')
    parser.add_argument('--easgd-beta', required=False, type=float, default=0.9,
                        help='beta parameter of easgd')

    args, unknown = parser.parse_known_args()
    logging.info(str(args))

    if args.dir_path == '':
        rom_name = os.path.splitext(os.path.basename(args.rom))[0]
        time_str = time.strftime("%m%d_%H%M_%S", time.localtime())
        args.dir_path = ('dqn-%s-%d_' % (rom_name,int(args.lr*10**5)))+time_str \
                        + "_" + os.environ.get('DMLC_TASK_ID')
    else:
        args.dir_path =  args.dir_path + "_" + os.environ.get('DMLC_TASK_ID')
    logging.info("saving to dir: "+args.dir_path)
    if args.ctx == None:
        args.ctx = os.environ.get('CTX')
    logging.info("Context: %s" % args.ctx)
    ctx = re.findall('([a-z]+)(\d*)', args.ctx)
    ctx = [(device, int(num)) if len(num) >0 else (device, 0) for device, num in ctx]

    # Async verision
    nactor= args.nactor
    param_update_period = args.param_update_period

    replay_start_size = args.replay_start_size
    max_start_nullops = 30
    replay_memory_size = args.replay_memory_size
    history_length = 4
    rows = 84
    cols = 84
    q_ctx = mx.Context(*ctx[0])
    games = []
    for g in range(nactor):
        games.append(AtariGame(rom_path=args.rom, resize_mode=args.resize_mode, replay_start_size=replay_start_size,
                             resized_rows=rows, resized_cols=cols, max_null_op=max_start_nullops,
                             replay_memory_size=replay_memory_size, display_screen=args.visualization,
                             history_length=history_length))


    ##RUN NATURE
    freeze_interval = 40000/nactor
    freeze_interval /= param_update_period
    epoch_num = args.epoch_num
    steps_per_epoch = 4000000/nactor
    discount = 0.99
    save_screens = False
    eps_start = numpy.ones((3,))* args.start_eps
    eps_min = numpy.array([0.1,0.01,0.5])
    eps_decay = (eps_start - eps_min) / (args.exploration_period/nactor)
    eps_curr = eps_start
    eps_id = numpy.zeros((nactor,))
    eps_update_period = args.eps_update_period
    eps_update_count = numpy.zeros((nactor,))

    single_batch_size = args.single_batch_size
    minibatch_size = nactor * single_batch_size
    action_num = len(games[0].action_set)
    data_shapes = {'data': (minibatch_size, history_length) + (rows, cols),
                   'dqn_action': (minibatch_size,), 'dqn_reward': (minibatch_size,)}

    if args.symbol == "nature":
        dqn_sym = dqn_sym_nature(action_num)
    elif args.symbol == "nips":
        dqn_sym = dqn_sym_nips(action_num)
    else:
        raise NotImplementedError
    qnet = Base(data_shapes=data_shapes, sym=dqn_sym, name='QNet',
                  initializer=DQNInitializer(factor_type="in"),
                  ctx=q_ctx)
    target_qnet = qnet.copy(name="TargetQNet", ctx=q_ctx)

    if args.optimizer == "adagrad":
        optimizer = mx.optimizer.create(name=args.optimizer, learning_rate=args.lr, eps=args.eps,
                        clip_gradient=args.clip_gradient,
                        rescale_grad=1.0, wd=args.wd)
    elif args.optimizer == "rmsprop" or args.optimizer == "rmspropnoncentered":
        optimizer = mx.optimizer.create(name=args.optimizer, learning_rate=args.lr, eps=args.eps,
                        clip_gradient=args.clip_gradient,gamma1=args.rms_decay,gamma2=0,
                        rescale_grad=1.0, wd=args.wd)
        lr_decay = (args.lr - 0)/(steps_per_epoch*epoch_num/param_update_period)
    weight_write_lock = Lock()
    # Create kvstore
    use_easgd = False
    if args.kv_type != None:
        kvType = args.kv_type
        kv = kvstore.create(kvType)
        #Initialize kvstore
        for idx,v in enumerate(qnet.params.values()):
            kv.init(idx,v)
        if args.server_optimizer == "easgd":
            use_easgd = True
            easgd_beta = args.easgd_beta
            easgd_alpha = easgd_beta/(args.kvstore_update_period*args.nworker)
            server_optimizer = mx.optimizer.create(name="ServerEasgd",learning_rate=easgd_alpha)
            easgd_eta = 0.00025
            central_weight = OrderedDict([(n, v.copyto(q_ctx))
                                            for n, v in qnet.params.items()])
            kv.set_optimizer(server_optimizer)
            updater = mx.optimizer.get_updater(optimizer)
            easgd_update_thread = EasgdThread(kv,central_weight,qnet.params,easgd_alpha,
                            weight_write_lock,args.easgd_update_period,args.param_update_period)
            easgd_update_thread.start()
        else:
            kv.set_optimizer(optimizer)
        kvstore_update_period = args.kvstore_update_period
        npy_rng = numpy.random.RandomState(123456+kv.rank)
    else:
        updater = mx.optimizer.get_updater(optimizer)

    qnet.print_stat()
    target_qnet.print_stat()

    states_buffer_for_act = numpy.zeros((nactor, history_length)+(rows, cols),dtype='uint8')
    states_buffer_for_train = numpy.zeros((minibatch_size, history_length+1)+(rows, cols),dtype='uint8')
    next_states_buffer_for_train = numpy.zeros((minibatch_size, history_length)+(rows, cols),dtype='uint8')
    actions_buffer_for_train = numpy.zeros((minibatch_size, ),dtype='uint8')
    rewards_buffer_for_train = numpy.zeros((minibatch_size, ),dtype='float32')
    terminate_flags_buffer_for_train = numpy.zeros((minibatch_size, ),dtype='bool')
    # Begin Playing Game
    training_steps = 0
    total_steps = 0
    ave_fps = 0
    ave_loss = 0
    time_for_info = time.time()
    parallel_executor = concurrent.futures.ThreadPoolExecutor(nactor)


    for epoch in xrange(epoch_num):
        # Run Epoch
        steps_left = steps_per_epoch
        episode = 0
        epoch_reward = 0
        start = time.time()
        #
        for g,game in enumerate(games):
            game.start()
            game.begin_episode()
            eps_rand = npy_rng.rand()
            if eps_rand<0.4:
                eps_id[g] = 0
            elif eps_rand<0.7:
                eps_id[g] = 1
            else:
                eps_id[g] = 2
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
                                   ave_fps, (eps_curr[eps_id[g]]))
                    info_str += ", Avg Loss:%f" % ave_loss
                    if episode_stats[g].episode_action_step > 0:
                        info_str += ", Avg Q Value:%f/%d" % (episode_stats[g].episode_q_value / episode_stats[g].episode_action_step,
                                                          episode_stats[g].episode_action_step)
                    if g == 0: logging.info(info_str)
                    if eps_update_count[g] * eps_update_period < total_steps:
                        eps_rand = npy_rng.rand()
                        if eps_rand<0.4:
                            eps_id[g] = 0
                        elif eps_rand<0.7:
                            eps_id[g] = 1
                        else:
                            eps_id[g] = 2
                        eps_update_count[g] += 1
                    game.begin_episode(steps_left)
                    episode_stats[g] = EpisodeStat()

            if total_steps > history_length:
                for g, game in enumerate(games):
                    current_state = game.current_state()
                    states_buffer_for_act[g] = current_state

            states = nd.array(states_buffer_for_act,ctx=q_ctx) / float(255.0)
            with weight_write_lock:
                qval_npy = qnet.forward(batch_size=nactor, data=states)[0].asnumpy()
            actions_that_max_q = numpy.argmax(qval_npy,axis=1)
            actions = [0]*nactor
            for g, game in enumerate(games):
                # 1. We need to choose a new action based on the current game status
                if games[g].state_enabled and games[g].replay_memory.sample_enabled:
                    do_exploration = (npy_rng.rand() < eps_curr[eps_id[g]])
                    if do_exploration:
                        action = npy_rng.randint(action_num)
                    else:
                        # TODO Here we can in fact play multiple gaming instances simultaneously and make actions for each
                        # We can simply stack the current_state() of gaming instances and give prediction for all of them
                        # We need to wait after calling calc_score(.), which makes the program slow
                        # TODO Profiling the speed of this part!
                        action = actions_that_max_q[g]
                        episode_stats[g].episode_q_value += qval_npy[g, action]
                        episode_stats[g].episode_action_step += 1
                else:
                    action = npy_rng.randint(action_num)
                actions[g] = action
            # t0=time.time()
            for ret in parallel_executor.map(play_game, zip(games, actions)):
                pass
            # t1=time.time()
            # logging.info("play time: %f" % (t1-t0))
            eps_curr = numpy.maximum(eps_curr - eps_decay, eps_min)
            total_steps += 1
            steps_left -= 1
            if total_steps % 100 == 0:
                this_time = time.time()
                ave_fps = (100/(this_time-time_for_info))
                time_for_info = this_time


            # 3. Update our Q network if we can start sampling from the replay memory
            #    Also, we update every `update_interval`
            if total_steps > minibatch_size and \
                total_steps % (param_update_period) == 0 and \
                games[-1].replay_memory.sample_enabled:
                # 3.1 Draw sample from the replay_memory
                for g,game in enumerate(games):
                    episode_stats[g].episode_update_step += 1
                    nsample = single_batch_size
                    i0 = (g*nsample)
                    i1 = (g+1)*nsample
                    if args.sample_policy == "recent":
                        action, reward, terminate_flag=game.replay_memory.sample_last(batch_size=nsample,\
                            states=states_buffer_for_train,offset=i0)
                    elif args.sample_policy == "random":
                        action, reward, terminate_flag=game.replay_memory.sample_inplace(batch_size=nsample,\
                            states=states_buffer_for_train,offset=i0)
                    actions_buffer_for_train[i0:i1]= action
                    rewards_buffer_for_train[i0:i1]= reward
                    terminate_flags_buffer_for_train[i0:i1]=terminate_flag
                states = nd.array(states_buffer_for_train[:,:-1], ctx=q_ctx) / float(255.0)
                next_states = nd.array(states_buffer_for_train[:,1:], ctx=q_ctx) / float(255.0)
                actions = nd.array(actions_buffer_for_train, ctx=q_ctx)
                rewards = nd.array(rewards_buffer_for_train, ctx=q_ctx)
                terminate_flags = nd.array(terminate_flags_buffer_for_train, ctx=q_ctx)
                with weight_write_lock:
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

                    if args.kv_type == None or use_easgd:
                        qnet.update(updater=updater)
                    else:
                        update_on_kvstore(kv, qnet.params, qnet.params_grad)

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
                    with weight_write_lock:
                        qnet.copy_params_to(target_qnet)

                if args.optimizer == "rmsprop" or args.optimizer == "rmspropnoncentered":
                    optimizer.lr -= lr_decay

                if save_screens and training_steps % (60*60*2/param_update_period) == 0:
                    logging.info("saving screenshots")
                    for g in range(nactor):
                        screen = states_buffer_for_train[(g*single_batch_size),-2,:,:].reshape(
                                                            states_buffer_for_train.shape[2:])
                        cv2.imwrite("screen_"+str(g)+".png",screen)
                training_steps += 1





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

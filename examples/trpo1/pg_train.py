import os

import numpy as np
from arena.games.gym_wrapper import GymWrapper
from arena.games.complex_wrapper import ComplexWrapper
from arena.experiment import Experiment
import gym
from gym.spaces import Box, Discrete
import argparse
from batch_agent import BatchUpdateAgent
from async_agent import AsyncAgent
from hrl_agent import HrlAgent
from flexible_hrl_agent import FlexibleHrlAgent
import logging
from custom_ant import CustomAnt
from gather_env import GatherEnv
from maze_env import MazeEnv
from custom_pend import CustomPend
from single_gather_env import SingleGatherEnv, SimpleSingleGatherEnv
import sys
import os
from collections import OrderedDict
from tf_utils import aggregate_feature, concat_feature, concat_without_task, str2bool
import subprocess

BATH_SIZE = 10000


# np.set_printoptions(precision=4)

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])


def linear_moving_value(x1, x2, t1, t2, t):
    if t < t1:
        return x1
    if t > t2:
        return x2
    return x1 + (t - t1) * (x2 - x1) / (t2 - t1)


def exp_moving_value(x1, x2, t1, t2, t):
    if t < t1:
        return x1
    if t > t2:
        return x2
    return x1 * (x2 / x1) ** ((t - t1) / (t2 - t1))


DIRECTIONS = np.array([(1., 0), (0, 1.), (-1., 0), (0, -1.)])


def x_forward_obj():
    return np.array((1, 0))


def x_backward_obj():
    return np.array((-1, 0))


def x_up_obj():
    return np.array((0, 1))


def x_down_obj():
    return np.array((0, -1))


def v_11():
    return np.sqrt(1 / 2) * np.array((1, 1))


def v_1n1():
    return np.sqrt(1 / 2) * np.array((1, -1))


def v_n11():
    return np.sqrt(1 / 2) * np.array((-1, 1))


def v_n1n1():
    return np.sqrt(1 / 2) * np.array((-1, -1))


def random_direction():
    choice = np.random.randint(0, 4)
    return DIRECTIONS[choice, :]


def up_for():
    choice = np.random.randint(0, 2)
    return DIRECTIONS[choice, :]

def random_cont_direction():
    a = np.random.rand() * 2 * np.pi
    return np.array([np.cos(a), np.sin(a)])


def x_for_back():
    choice = np.random.randint(0, 2) * 2
    return DIRECTIONS[choice, :]


cwd = os.getcwd()

def main():
    parser = argparse.ArgumentParser(description='Script to test the network on cartpole swingup.')
    parser.add_argument('--lr', required=False, default=0.0001, type=float,
                        help='learning rate of the choosen optimizer')
    parser.add_argument('--vlr', required=False, default=0.0003, type=float,
                        help='learning rate of the critic')
    parser.add_argument('--clip-gradient', default=True, type=bool, help='whether to clip the gradient')
    parser.add_argument('--gpu', required=False, type=int, default=0,
                        help='Running Context.')
    parser.add_argument('--nactor', required=False, type=int, default=20,
                        help='Number of parallel actor-learners')
    parser.add_argument('--batch-size', required=False, type=int, default=BATH_SIZE,
                        help='batch size')
    parser.add_argument('--num-steps', required=False, type=int, default=15e7,
                        help='Total number of steps')
    parser.add_argument('--switcher-length', required=False, type=int, default=10,
                        help='switcher length')
    parser.add_argument('--lr-decrease', default=True, type=bool, help='whether to decrease lr')
    parser.add_argument('--batch-mode', required=False, type=str, default='timestep',
                        help='timestep or episode')
    parser.add_argument('--kl', required=False, default=None, type=float,
                        help='target kl')
    parser.add_argument('--ent-k', required=False, default=0, type=float,
                        help='entropy loss weight')
    parser.add_argument('--lam', required=False, default=0.97, type=float,
                        help='gae lambda')
    parser.add_argument('--gamma', required=False, default=0.995, type=float,
                        help='gae lambda')
    parser.add_argument('--withimg', default=False, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--load-model', default=False, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--norm-gae', default=True, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--load-dir', default="models", type=str, help='model directory')
    parser.add_argument('--load-leaf', default=True, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--reset-exp', default=False, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--no-train', default=False, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--train-leaf', default=False, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--env', default="ant", type=str, help='env')
    parser.add_argument('--loss', default="PPO", type=str, help='loss')
    parser.add_argument('--rl-method', default="ACKTR", type=str, help='rl method')
    parser.add_argument('--npret', required=False, type=int, default=-1,
                        help='num pretrain')
    parser.add_argument('--nfeat', required=False, type=int, default=0,
                        help='num img feat')
    parser.add_argument('--save-model', required=False, type=int, default=10,
                        help='save_model')

    parser.add_argument('--render', default="off", type=str, help='rendoer option')
    parser.add_argument("--debug", default=False, type=str2bool, nargs='?', const=True, help='debug')
    args = parser.parse_args()

    should_profile = False

    # Each trajectory will have at most 1000 time steps
    T = 1000
    num_actors = args.nactor
    steps_per_epoch = args.batch_size
    num_epoch = int(args.num_steps / steps_per_epoch)
    num_updates = int(args.num_steps / (args.batch_size * 100))
    # final_factor = 0.01
    test_length = 0

    import multiprocessing
    render_lock = multiprocessing.Lock()
    barrier = multiprocessing.Barrier(num_actors)


    append_image = args.withimg
    feat_sup = False

    hrl0 = OrderedDict(move1d=x_for_back, move0=x_forward_obj, move1=x_backward_obj)
    hrl_8d = OrderedDict(move0=x_forward_obj, move1=x_backward_obj,
                         move2=x_up_obj, move3=x_down_obj,
                         move4=v_11, move5=v_n1n1, move6=v_1n1, move7=v_n11)

    hrl_move2d = OrderedDict(move2d=random_direction,
                             move0=x_forward_obj, move1=x_backward_obj,
                             move2=x_up_obj, move3=x_down_obj
                             )
    hrl2 = OrderedDict(reach2d=random_direction,
                       move0=x_forward_obj, move1=x_backward_obj,
                       move2=x_up_obj, move3=x_down_obj
                       )
    hrl_dimage = OrderedDict(moves2d=random_direction,
                             moves0=x_forward_obj, moves1=x_backward_obj,
                             moves2=x_up_obj, moves3=x_down_obj
                             )
    hrl_changing_goal = OrderedDict(dynamic2d=random_direction,
                                    move0=x_forward_obj, move1=x_backward_obj,
                                    move2=x_up_obj, move3=x_down_obj
                                    )
    hrl_c1 = OrderedDict(reachc1=random_direction,
                         move0=x_forward_obj, move1=x_backward_obj,
                         move2=x_up_obj, move3=x_down_obj
                         )
    hrl_c05 = OrderedDict(reachc05=random_direction,
                          move0=x_forward_obj, move1=x_backward_obj,
                          move2=x_up_obj, move3=x_down_obj
                          )
    hrl_fake = OrderedDict(**{"cartpole_hrl": "", "CartPole-v1_0": "", "CartPole-v1_1": ""})
    hrl_up_for = OrderedDict(move_up_for=up_for, move0=x_forward_obj, move2=x_up_obj)
    hrl_simple1d = OrderedDict(simplehrl1d=x_for_back, move0=x_forward_obj, move1=x_backward_obj)
    hrl_root_tasks = dict(move1d=hrl0, move2d=hrl_move2d, reach2d=hrl2, dynamic2d=hrl_changing_goal,
                          reachc1=hrl_c1, reachc05=hrl_c05, moves2d=hrl_dimage, cartpole_hrl=hrl_fake,
                          move_up_for=hrl_up_for, simplehrl1d=hrl_simple1d)

    full_tasks = [args.env]
    if args.env in hrl_root_tasks:
        full_tasks = hrl_root_tasks[args.env]

    def f_create_env(render_lock=None, pid=0):
        # env = GatherEnv()
        if args.env == "single":
            with_state_task = False
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                  f_gen_obj=random_cont_direction,
                                  reset_goal_prob=0, )

        elif args.env == "custant":
            env = CustomAnt(file_path=cwd + "/cust_ant.xml")
        elif args.env in hrl_8d:
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=hrl_8d[args.env],
                                  reset_goal_prob=0, )
        elif args.env == "forward_and_backward":
            with_state_task = not (append_image and not feat_sup)
            f_direction = x_forward_obj if pid % 2 == 0 else x_backward_obj
            logging.info("actuator[{}], direction: {}".format(pid, f_direction))
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                  f_gen_obj=f_direction)

        elif args.env in hrl0:
            with_state_task = False
            if args.env == list(hrl_root_tasks[args.env].keys())[0]:
                use_internal_reward = False
            else:
                use_internal_reward = True
            subtask_dirs = np.stack([v() for (k, v) in list(hrl0.items())[1:]], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                  f_gen_obj=hrl0[args.env],  # x_forward_obj if pid % 2 else x_backward_obj,#
                                  reset_goal_prob=0,
                                  use_internal_reward=use_internal_reward,
                                  subtask_dirs=subtask_dirs)
        elif args.env == "simplehrl1d":
            with_state_task = False
            if args.env == list(hrl_root_tasks[args.env].keys())[0]:
                use_internal_reward = False
            else:
                use_internal_reward = True
            subtask_dirs = np.stack([v() for (k, v) in list(hrl0.items())[1:]], axis=0)
            env = SimpleSingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                        f_gen_obj=x_for_back,  # x_forward_obj if pid % 2 else x_backward_obj,#
                                        reset_goal_prob=0,
                                        use_internal_reward=use_internal_reward,
                                        subtask_dirs=subtask_dirs)
        elif args.env in hrl_move2d:
            subtask_dirs = np.stack([v() for (k, v) in list(hrl_move2d.items())[1:]], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=hrl_move2d[args.env],
                                  reset_goal_prob=0.0,
                                  subtask_dirs=subtask_dirs)

        elif args.env in hrl_changing_goal:
            with_state_task = False

            subtask_dirs = np.stack([v() for (k, v) in list(hrl_changing_goal.items())[1:]], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                  f_gen_obj=hrl_changing_goal[args.env],
                                  reset_goal_prob=0.01,
                                  subtask_dirs=subtask_dirs)
        elif args.env == "reach_test":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_direction,
                                  use_sparse_reward=True)
        elif args.env == "reach2d":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_direction,
                                  use_sparse_reward=True, )
        elif args.env == "reachc1":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_direction,
                                  catch_range=1,
                                  obj_dist=1.25,
                                  use_sparse_reward=True, )
        elif args.env == "reachc05":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_direction,
                                  catch_range=0.5,
                                  obj_dist=1.25,
                                  use_sparse_reward=True, )
        elif args.env in hrl_dimage:
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=hrl_dimage[args.env],
                                  reset_goal_prob=0, )
        elif args.env == "movetest":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=x_for_back,
                                  forward_scale=1.0,
                                  reset_goal_prob=0, )
        elif args.env == "simplemove1d":
            env = SimpleSingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                        f_gen_obj=x_for_back,
                                        forward_scale=10.0,
                                        reset_goal_prob=0, )
        elif args.env == "scalemove1d":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=x_for_back,
                                  forward_scale=10.0,
                                  reset_goal_prob=0, )
        elif args.env == "cartpole_hrl":
            env = gym.make("CartPole-v1")
        elif args.env in hrl_up_for:
            subtask_dirs = np.stack([v() for (k, v) in list(hrl_root_tasks[args.env].items())[1:]], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=hrl_root_tasks[args.env][args.env],
                                  reset_goal_prob=0, subtask_dirs=subtask_dirs)
        else:
            env = gym.make(args.env)

        dummy_image = False
        final_env = ComplexWrapper(env, max_episode_length=T,
                                   append_image=append_image, rgb_to_gray=True,
                                   visible_state_ids=range(env.observation_space.shape[0]),
                                   num_frame=1,
                                   dummy_image=dummy_image,
                                   render_lock=render_lock)
        logging.info("created env")
        return final_env
        # env = CustomPend()
        # return ComplexWrapper(env, max_episode_length=T,
        #                       append_image=True, new_img_size=(84, 84), rgb_to_gray=True,
        #                       visible_state_ids=np.array((True, True, True, True)),
        #                       s_transform=ident,
        #                       num_frame=3)

    def f_create_agent(observation_space, action_space,
                       shared_params, stats_rx, acts_tx,
                       is_learning, global_t, pid):
        if len(full_tasks) == 1:
            return BatchUpdateAgent(
                observation_space, action_space,
                shared_params, stats_rx, acts_tx,
                is_learning, global_t, pid,
            )
        else:
            # return HrlAgent(
            #     observation_space, action_space,
            #     shared_params, stats_rx, acts_tx,
            #     is_learning, global_t, pid,
            #     full_tasks=full_tasks
            # )
            return FlexibleHrlAgent(
                observation_space, action_space,
                shared_params, stats_rx, acts_tx,
                is_learning, global_t, pid,
                full_tasks=full_tasks
            )

    def const_batch_size(n_update):
        return args.batch_size

    def const_kl(n_update):
        return args.kl

    if args.kl is None:
        f_target_kl = None
    else:
        f_target_kl = const_kl

    start_t = 0
    end_t = args.num_steps / 10000

    def get_batch_size(n_update):
        b1 = 2000 / num_actors
        b2 = 20000 / num_actors
        b = round(linear_moving_value(b1, b2, start_t, end_t, n_update))
        return num_actors * b

    def get_target_kl(n_update):
        k1 = 0.003
        k2 = 0.0001
        k = exp_moving_value(k1, k2, start_t, end_t, n_update)
        return k

    def create_session():
        import tensorflow as tf
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        if args.debug:
            from tensorflow.python import debug as tf_debug
            sess_debug = tf_debug.LocalCLIDebugWrapperSession(sess)
            return sess_debug
        else:
            return sess

    def pg_shared_params():
        from policy_gradient_model import PolicyGradientModel
        from dict_memory import DictMemory
        sample_env = f_create_env()
        observation_space = sample_env.observation_space
        action_space = sample_env.action_space
        sample_env.env.close()
        n_imgfeat = args.nfeat if append_image else 0
        comb_methd = concat_feature if append_image else aggregate_feature

        comb_methd = concat_without_task if feat_sup else comb_methd
        session = create_session()

        model = PolicyGradientModel(observation_space, action_space,
                                    name=args.env,
                                    timestep_limit=T,
                                    num_actors=num_actors,
                                    f_batch_size=const_batch_size,
                                    batch_mode=args.batch_mode,
                                    f_target_kl=f_target_kl,
                                    lr=args.lr,
                                    critic_lr=args.vlr,
                                    n_imgfeat=n_imgfeat,
                                    mode=args.rl_method,
                                    update_per_epoch=4,
                                    kl_history_length=1,
                                    comb_method=comb_methd,
                                    surr_loss=args.loss,
                                    ent_k=args.ent_k,
                                    session=session,
                                    load_old_model=args.load_model,
                                    reset_exp=args.reset_exp,
                                    model_load_dir=args.load_dir,
                                    parallel_predict=True,
                                    should_train=not args.no_train,
                                    save_model=args.save_model)
        memory = DictMemory(gamma=args.gamma, lam=args.lam, normalize=True,
                            timestep_limit=T,
                            f_critic={"decider": model.compute_critic},
                            num_actors=num_actors,
                            f_check_batch=model.check_batch_finished)
        return {"models": [model], "memory": memory}

    def flexible_hrl_shared_params():
        from policy_gradient_model import PolicyGradientModel
        from dict_memory import DictMemory
        sample_env = f_create_env()
        observation_space = sample_env.observation_space
        action_space = sample_env.action_space
        sample_env.env.close()
        n_imgfeat = args.nfeat if append_image else 0
        comb_methd = concat_feature if append_image else aggregate_feature

        comb_methd = concat_without_task if feat_sup else comb_methd
        session = create_session()

        decider_action_space = Discrete(len(full_tasks) - 1)
        decider_observation_space = observation_space

        # Data structure: [state, image, current policy, current execution time, is initial step
        switcher_action_space = Discrete(2)
        switcher_obseravation_space = [*observation_space,
                                       Box(np.array([0, 0, 0]),
                                           np.array([decider_action_space.n, np.inf, 1.0])
                                           )]

        def f_train_root(n):
            return n > args.npret

        def f_train_leaf(n):
            return not f_train_root(n)

        num_leaf = (len(full_tasks.keys()) - 1)

        def hrl_batch_size(n_update):
            return args.batch_size if n_update > args.npret else args.batch_size * num_leaf

        decider_model = PolicyGradientModel(decider_observation_space, decider_action_space,
                                            name=args.env,
                                            timestep_limit=T,
                                            num_actors=num_actors,
                                            f_batch_size=hrl_batch_size,
                                            batch_mode=args.batch_mode,
                                            f_target_kl=f_target_kl,
                                            lr=args.lr,
                                            critic_lr=args.vlr,
                                            n_imgfeat=n_imgfeat,
                                            mode=args.rl_method,
                                            kl_history_length=1,
                                            comb_method=comb_methd,
                                            surr_loss=args.loss,
                                            ent_k=args.ent_k,
                                            session=session,
                                            load_old_model=args.load_model,
                                            model_load_dir=args.load_dir,
                                            should_train=not args.no_train,
                                            f_train_this_epoch=f_train_root,
                                            parallel_predict=False,
                                            save_model=args.save_model,
                                            is_switcher_with_init_len=False,
                                            is_decider=True)
        switcher_model = PolicyGradientModel(switcher_obseravation_space, switcher_action_space,
                                             name=args.env,
                                             timestep_limit=T,
                                             num_actors=num_actors,
                                             f_batch_size=hrl_batch_size,
                                             batch_mode=args.batch_mode,
                                             f_target_kl=f_target_kl,
                                             lr=args.lr,
                                             critic_lr=args.vlr,
                                             n_imgfeat=n_imgfeat,
                                             mode=args.rl_method,
                                             kl_history_length=1,
                                             comb_method=comb_methd,
                                             surr_loss=args.loss,
                                             ent_k=args.ent_k,
                                             session=session,
                                             load_old_model=False,
                                             model_load_dir=args.load_dir,
                                             should_train=False,
                                             f_train_this_epoch=f_train_root,
                                             parallel_predict=False,
                                             save_model=args.save_model,
                                             is_switcher_with_init_len=args.switcher_length)
        models = {"decider": decider_model, "switcher": switcher_model, "leafs": []}

        for i, env_name in enumerate(list(full_tasks.keys())[1:]):
            if args.env in hrl_fake:
                const_action = i
            else:
                const_action = None
            p = PolicyGradientModel(observation_space, action_space,
                                    name=env_name,
                                    timestep_limit=T,
                                    num_actors=num_actors,
                                    n_imgfeat=n_imgfeat,
                                    comb_method=comb_methd,
                                    ent_k=args.ent_k,
                                    session=session,
                                    load_old_model=args.load_leaf,
                                    model_load_dir=args.load_dir,
                                    reset_exp=args.reset_exp,
                                    should_train=args.train_leaf,
                                    f_train_this_epoch=f_train_leaf,
                                    parallel_predict=False,
                                    f_batch_size=const_batch_size,
                                    batch_mode=args.batch_mode,
                                    f_target_kl=f_target_kl,
                                    lr=args.lr,
                                    critic_lr=args.vlr,
                                    mode=args.rl_method,
                                    kl_history_length=1,
                                    surr_loss=args.loss,
                                    save_model=args.save_model,
                                    is_switcher_with_init_len=False,
                                    const_action=const_action
                                    )
            models["leafs"].append(p)
        # for p in models[1:]:
        #     p.restore_parameters()
        # memory = DictMemory(gamma=args.gamma, lam=args.lam, normalize=True,
        #                     timestep_limit=T,
        #                     f_critic=root_model.compute_critic,
        #                     num_actors=num_actors,
        #                     f_check_batch=root_model.check_batch_finished, )
        memory = DictMemory(gamma=args.gamma, lam=args.lam, normalize=args.norm_gae,
                            timestep_limit=T,
                            f_critic={"decider": decider_model.compute_critic,
                                      "switcher": switcher_model.compute_critic,
                                      "leafs": [m.compute_critic for m in models["leafs"]]
                                      },
                            num_leafs=len(full_tasks) - 1,
                            num_actors=num_actors,
                            f_check_batch=decider_model.check_batch_finished, )

        return {"models": models, "memory": memory}

    f_create_params = pg_shared_params if len(full_tasks) == 1 else flexible_hrl_shared_params

    single_process_mode = True  # True if append_image else False
    experiment = Experiment(f_create_env, f_create_agent,
                            f_create_params, single_process_mode=single_process_mode, render_option=args.render,
                            log_episodes=True)
    logging.info("run arges: {}".format(args))
    logging.info("version: {}".format(str(get_git_revision_short_hash())))

    experiment.run_parallelly(num_actors, num_epoch, steps_per_epoch,
                              with_testing_length=test_length)


if __name__ == '__main__':
    main()

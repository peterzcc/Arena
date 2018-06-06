import multiprocessing as mp

import numpy as np
from arena.experiment import Experiment
from gym.spaces import Box, Discrete
import argparse

import logging
import os

import subprocess
from arena.games.cust_control.env_library import make_env

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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

cwd = os.getcwd()

def main():
    parser = argparse.ArgumentParser(description='Script to test the network on cartpole swingup.')
    parser.add_argument('--lr', required=False, default=0.001, type=float,
                        help='learning rate of the choosen optimizer')
    parser.add_argument('--npass', required=False, type=int, default=1,
                        help='num pass')
    parser.add_argument('--multi-update', default=False, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--minibatch-size', required=False, type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--vlr', required=False, default=0.0003, type=float,
                        help='learning rate of the critic')
    parser.add_argument('--clip-gradient', default=True, type=bool, help='whether to clip the gradient')
    parser.add_argument('--gpu', required=False, type=int, default=0,
                        help='Running Context.')
    parser.add_argument('--nactor', required=False, type=int, default=20,
                        help='Number of parallel actor-learners')
    parser.add_argument('--batch-size', required=False, type=int, default=BATH_SIZE,
                        help='batch size')
    parser.add_argument('--num-steps', required=False, type=int, default=32e7,
                        help='Total number of steps')
    parser.add_argument('--switcher-length', required=False, type=int, default=10,
                        help='switcher length')
    parser.add_argument('--switcher-start', required=False, type=int, default=0,
                        help='switcher training start time')
    parser.add_argument('--lr-decrease', default=True, type=bool, help='whether to decrease lr')
    parser.add_argument('--batch-mode', required=False, type=str, default='timestep',
                        help='timestep or episode')
    parser.add_argument('--kl', required=False, default=None, type=float,
                        help='target kl')
    parser.add_argument('--ent-k', required=False, default=0, type=float,
                        help='entropy loss weight')
    parser.add_argument('--switcher-k', required=False, default=0.01, type=float,
                        help='entropy loss weight')
    parser.add_argument('--lam', required=False, default=0.97, type=float,
                        help='gae lambda')
    parser.add_argument('--gamma', required=False, default=0.995, type=float,
                        help='gae lambda')
    parser.add_argument('--stdbias', required=False, default=0.0, type=float,
                        help='policy std bias')

    parser.add_argument('--stddev', required=False, default=1.0, type=float,
                        help='policy logstd sample std')
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
    parser.add_argument('--train-decider', default=True, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--train-switcher', default=False, type=str2bool, nargs='?',
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
    parser.add_argument('--decrease-with-initial', required=False, default=None, type=float,
                        help='initial wass')
    parser.add_argument('--wass-decrease-period', required=False, default=15.e6, type=float,
                        help='wass decrease period')
    parser.add_argument('--use-mix', default=False, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--normalize-wass', default=False, type=str2bool, nargs='?',
                        const=True, )

    parser.add_argument('--render', default="off", type=str, help='rendoer option')
    parser.add_argument("--savestats", default=False, type=str2bool, nargs='?', const=True, help='savestats')
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

    append_image = args.withimg

    _, env_info = make_env(args.env, args.withimg, T)
    env_args = dict(env_name=args.env, withimg=args.withimg, T=T)
    full_tasks = env_info["full_tasks"]
    is_fake_hrl = env_info["is_fake_hrl"]

    def f_create_agent(observation_space, action_space,
                       shared_params, stats_rx, acts_tx,
                       is_learning, global_t, pid):
        from batch_agent import BatchUpdateAgent
        from flexible_hrl_agent import FlexibleHrlAgent
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
    elif args.decrease_with_initial is not None:

        initial_kl = args.decrease_with_initial
        final_kl = args.kl
        decreasing_period_t_step = args.wass_decrease_period
        decreasingg_period_n = decreasing_period_t_step / args.batch_size
        log_power = (1.0 / decreasingg_period_n) * np.log(final_kl / initial_kl)
        logging.warning(
            "Using decreasing learning rate, from {} to {} in {} steps".format(
                initial_kl,
                final_kl,
                decreasingg_period_n
            ))

        def exp_decreasing_kl(n_update):
            this_kl = initial_kl * np.exp(log_power * np.minimum(n_update, decreasingg_period_n))
            return this_kl

        f_target_kl = exp_decreasing_kl
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
        gpu_options = tf.GPUOptions(allow_growth=True)  # TODO per_process_gpu_memory_fraction=1250 / 8000)  #
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False,
                                                allow_soft_placement=True))
        if args.debug:
            from tensorflow.python import debug as tf_debug
            sess_debug = tf_debug.LocalCLIDebugWrapperSession(sess)
            return sess_debug
        else:
            return sess

    policy_shared_params = dict(npass=args.npass,
                                minibatch_size=args.minibatch_size,
                                multi_update=args.multi_update,
                                savestats=args.savestats,
                                policy_logstd_grad_bias=args.stdbias,
                                logstd_sample_dev=args.stddev,
                                use_mix=args.use_mix,
                                normalize_wass=args.normalize_wass)

    def pg_shared_params():
        from policy_gradient_model import PolicyGradientModel
        from dict_memory import DictMemory
        from tf_utils import aggregate_feature, concat_feature
        sample_env, env_info = make_env(**env_args)
        observation_space = sample_env.observation_space
        action_space = sample_env.action_space
        sample_env.env.close()
        n_imgfeat = args.nfeat if append_image else 0
        comb_methd = concat_feature if append_image else aggregate_feature

        # comb_methd = concat_without_task if feat_sup else comb_methd
        session = create_session()

        model = PolicyGradientModel(observation_space, action_space,
                                    name=args.env,
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
                                    loss_type=args.loss,
                                    ent_k=args.ent_k,
                                    session=session,
                                    load_old_model=args.load_model,
                                    reset_exp=args.reset_exp,
                                    model_load_dir=args.load_dir,
                                    parallel_predict=True,
                                    should_train=args.train_decider,
                                    save_model=args.save_model,
                                    **policy_shared_params)
        memory = DictMemory(gamma=args.gamma, lam=args.lam, normalize=args.norm_gae,
                            timestep_limit=T,
                            f_critic={"decider": model.compute_critic},
                            num_actors=num_actors,
                            f_check_batch=model.check_batch_finished)
        return {"models": [model], "memory": memory}

    def flexible_hrl_shared_params():
        from policy_gradient_model import PolicyGradientModel
        from dict_memory import DictMemory
        import tensorflow as tf
        from tf_utils import aggregate_feature, concat_feature
        sample_env, _ = make_env(**env_args)
        observation_space = sample_env.observation_space
        action_space = sample_env.action_space
        sample_env.env.close()
        n_imgfeat = args.nfeat if append_image else 0
        comb_methd = concat_feature if append_image else aggregate_feature

        #comb_methd = concat_without_task if feat_sup else comb_methd
        session = create_session()
        from gpu_manager import GpuManager
        GpuManager.find_available_gpus()

        decider_action_space = Discrete(len(full_tasks) - 1)
        decider_observation_space = observation_space

        # Data structure: [state, image, current policy, current execution time, is initial step
        switcher_action_space = Discrete(2)
        switcher_obseravation_space = [*observation_space,
                                       Box(np.array([0, 0, 0]),
                                           np.array([decider_action_space.n, np.inf, 1.0])
                                           )]

        def f_train_decider(n):
            return n > args.npret

        def f_train_switcher(n):
            return n > args.npret and n >= args.switcher_start

        def f_train_leaf(n):
            return not f_train_decider(n)

        num_leaf = (len(full_tasks.keys()) - 1)

        def hrl_batch_size(n_update):
            return args.batch_size if n_update > args.npret else args.batch_size * num_leaf

        with tf.device(GpuManager.get_a_gpu_and_change()):
            decider_model = PolicyGradientModel(decider_observation_space, decider_action_space,
                                                name=args.env,
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
                                                loss_type=args.loss,
                                                ent_k=args.ent_k,
                                                session=session,
                                                load_old_model=args.load_model,
                                                model_load_dir=args.load_dir,
                                                should_train=args.train_decider,
                                                f_train_this_epoch=f_train_decider,
                                                parallel_predict=False,
                                                save_model=args.save_model,
                                                is_switcher_with_init_len=False,
                                                is_decider=True,
                                                **policy_shared_params)
        with tf.device(GpuManager.get_a_gpu_and_change()):
            switcher_model = PolicyGradientModel(switcher_obseravation_space, switcher_action_space,
                                                 name=args.env,
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
                                                 loss_type=args.loss,
                                                 ent_k=args.ent_k,
                                                 session=session,
                                                 load_old_model=args.load_model,
                                                 model_load_dir=args.load_dir,
                                                 should_train=args.train_switcher,
                                                 f_train_this_epoch=f_train_switcher,
                                                 parallel_predict=False,
                                                 save_model=args.save_model,
                                                 is_switcher_with_init_len=args.switcher_length,
                                                 switcher_cost_k=args.switcher_k,
                                                 **policy_shared_params)
        models = {"decider": decider_model, "switcher": switcher_model, "leafs": []}

        for i, env_name in enumerate(list(full_tasks.keys())[1:]):
            if is_fake_hrl:
                const_action = i
            else:
                const_action = None
            with tf.device(GpuManager.get_a_gpu_and_change()):
                p = PolicyGradientModel(observation_space, action_space,
                                        name=env_name,
                                        num_actors=num_actors,
                                        n_imgfeat=30,  # MARK: manually set 0,  #
                                        comb_method=comb_methd,
                                        ent_k=args.ent_k,
                                        session=session,
                                        load_old_model=args.load_leaf,
                                        conv_sizes=(((3, 3), 16, 2), ((3, 3), 16, 2), ((3, 3), 4, 2)),
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
                                        loss_type=args.loss,
                                        save_model=args.save_model,
                                        is_switcher_with_init_len=False,
                                        const_action=const_action,
                                        **policy_shared_params
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

    single_process_mode = False  # True if append_image else False
    experiment = Experiment(env_args, f_create_agent,
                            f_create_params, single_process_mode=single_process_mode, render_option=args.render,
                            log_episodes=True)
    logging.info("run arges: {}".format(args))
    logging.info("version: {}".format(str(get_git_revision_short_hash())))

    experiment.run_parallelly(num_actors, num_epoch, steps_per_epoch,
                              with_testing_length=test_length)


if __name__ == '__main__':
    main()

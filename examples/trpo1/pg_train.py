import os

import numpy as np
from arena.games.gym_wrapper import GymWrapper
from arena.games.complex_wrapper import ComplexWrapper
from arena.experiment import Experiment
import gym
from gym.spaces import Discrete
import argparse
from batch_agent import BatchUpdateAgent
from hrl_agent import HrlAgent
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

BATH_SIZE = 10000



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


def random_direction():
    choice = np.random.randint(0, 4)
    return DIRECTIONS[choice, :]


def random_cont_direction():
    a = np.random.rand() * 2 * np.pi
    return np.array([np.cos(a), np.sin(a)])


def x_for_back():
    choice = np.random.randint(0, 2) * 2
    return DIRECTIONS[choice, :]
def main():
    parser = argparse.ArgumentParser(description='Script to test the network on cartpole swingup.')
    parser.add_argument('--lr', required=False, default=0.0001, type=float,
                        help='learning rate of the choosen optimizer')
    parser.add_argument('--clip-gradient', default=True, type=bool, help='whether to clip the gradient')
    parser.add_argument('--save-model', default=False, type=bool, help='whether to save the final model')
    parser.add_argument('--gpu', required=False, type=int, default=0,
                        help='Running Context.')
    parser.add_argument('--nactor', required=False, type=int, default=20,
                        help='Number of parallel actor-learners')
    parser.add_argument('--batch-size', required=False, type=int, default=BATH_SIZE,
                        help='batch size')
    parser.add_argument('--num-steps', required=False, type=int, default=15e7,
                        help='Total number of steps')
    parser.add_argument('--lr-decrease', default=True, type=bool, help='whether to decrease lr')
    parser.add_argument('--batch-mode', required=False, type=str, default='timestep',
                        help='timestep or episode')
    parser.add_argument('--kl', required=False, default=0.002, type=float,
                        help='target kl')
    parser.add_argument('--ent-k', required=False, default=0, type=float,
                        help='entropy loss weight')
    parser.add_argument('--lam', required=False, default=0.97, type=float,
                        help='gae lambda')
    parser.add_argument('--gamma', required=False, default=0.995, type=float,
                        help='gae lambda')
    parser.add_argument('--withimg', default=True, type=bool, help='append image input')
    parser.add_argument('--load-model', default=False, type=str2bool, nargs='?',
                        const=True, )
    parser.add_argument('--env', default="ant", type=str, help='env')
    parser.add_argument('--nae', required=False, type=int, default=0,
                        help='num ae train')
    parser.add_argument('--nfeat', required=False, type=int, default=0,
                        help='num img feat')
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

    mean = np.array([0, 0, 0, 0])
    final_std = np.array([0.5, 0.5, 0.5, 0.5])
    final_n_batch = 25
    noise_k = 1.0 / final_n_batch
    def state_preprocess(x,t):
        y = x.copy()
        t_batch = t/BATH_SIZE
        ratio = \
            noise_k*t_batch if t_batch < final_n_batch else 1.0
        #logging.debug("current_noise_std: {}".format(current_std))
        noise = np.random.normal(loc=mean,scale=final_std)
        y[0:2] = (1-ratio)*y[0:2] + noise*ratio
        return y

    def eliminated_state(x, t):
        y = x.copy()
        t_batch = t / BATH_SIZE
        ratio = \
            noise_k * t_batch if t_batch < final_n_batch else 1.0
        # logging.debug("current_noise_std: {}".format(current_std))
        noise = mean
        y[0:2] = (1 - ratio) * y[0:2] + noise * ratio
        return y

    def dropout_state(x, t):
        y = x.copy()
        t_batch = t / BATH_SIZE
        ratio = \
            noise_k * t_batch if t_batch < final_n_batch else 1.0
        # logging.debug("current_noise_std: {}".format(current_std))
        is_removed = (np.random.random_sample(size=None) < ratio)
        y[0:2] = mean if is_removed else y[0:2]
        return y

    def zeroed_state(x, t):
        y = x.copy()
        y[0:2] = mean
        return y

    def const_noise(x, t):
        y = x.copy()
        noise = np.random.normal(loc=mean, scale=final_std)
        y += noise
        return y
    def ident(x,t):
        return x

    import multiprocessing
    render_lock = multiprocessing.Lock()
    barrier = multiprocessing.Barrier(num_actors)
    cwd = os.getcwd()

    append_image = args.withimg
    feat_sup = False



    hrl0 = OrderedDict(move1d=x_for_back, move0=x_forward_obj, move1=x_backward_obj)

    hrl1 = OrderedDict(move2d=random_cont_direction,
                       move0=x_forward_obj, move1=x_backward_obj,
                       move2=x_up_obj, move3=x_down_obj
                       )
    hrl_root_tasks = dict(move1d=hrl0, move2d=hrl1)

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
        elif args.env == "forward_and_backward":
            with_state_task = not (append_image and not feat_sup)
            f_direction = x_forward_obj if pid % 2 == 0 else x_backward_obj
            logging.info("actuator[{}], direction: {}".format(pid, f_direction))
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                  f_gen_obj=f_direction)
        elif args.env in hrl0:
            with_state_task = False
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                  f_gen_obj=hrl0[args.env],
                                  reset_goal_prob=0, )
        elif args.env in hrl1:
            with_state_task = False
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                  f_gen_obj=hrl1[args.env],
                                  reset_goal_prob=0, )
        elif args.env == "reach_test":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_cont_direction,
                                  use_sparse_reward=True,
                                  obj_dist=1.25)
        else:
            env = gym.make(args.env)
        # env = MazeEnv()

        # return GymWrapper(env,
        #                   max_null_op=0, max_episode_length=T)
        # env = CustomAnt(file_path=cwd + "/cust_ant.xml")

        # env = SimpleSingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
        #                             f_gen_obj=forward_backward)
        final_env = ComplexWrapper(env, max_episode_length=T,
                                   append_image=append_image, rgb_to_gray=True,
                                   s_transform=ident,
                                   visible_state_ids=range(env.observation_space.shape[0]),
                                   num_frame=1,
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
            return HrlAgent(
                observation_space, action_space,
                shared_params, stats_rx, acts_tx,
                is_learning, global_t, pid,
                full_tasks=full_tasks
            )

    def const_batch_size(n_update):
        return args.batch_size

    def const_target_kl(n_update):
        return args.kl

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
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

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
                                    f_target_kl=const_target_kl,
                                    n_imgfeat=n_imgfeat,
                                    mode="ACKTR",
                                    update_per_epoch=4,
                                    kl_history_length=1,
                                    comb_method=comb_methd,
                                    ent_k=args.ent_k,
                                    session=session,
                                    load_old_model=args.load_model,
                                    should_train=not args.load_model)
        memory = DictMemory(gamma=args.gamma, lam=args.lam, normalize=True,
                            timestep_limit=T,
                            f_critic=model.compute_critic,
                            num_actors=num_actors,
                            f_check_batch=model.check_batch_finished)
        return {"models": [model], "memory": memory}

    def f_should_return_root(source, t, target=None):
        return t >= 10

    def hrl_shared_params():
        # TODO: hrl
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

        root_action_space = Discrete(len(full_tasks) - 1)
        root_model = PolicyGradientModel(observation_space, root_action_space,
                                         name=args.env,
                                         timestep_limit=T,
                                         num_actors=num_actors,
                                         f_batch_size=const_batch_size,
                                         batch_mode=args.batch_mode,
                                         f_target_kl=const_target_kl,
                                         n_imgfeat=n_imgfeat,
                                         mode="ACKTR",
                                         kl_history_length=1,
                                         comb_method=comb_methd,
                                         ent_k=args.ent_k,
                                         session=session,
                                         load_old_model=False,
                                         should_train=True,
                                         parallel_predict=False)
        models = [root_model]
        for env_name, _ in list(full_tasks.items())[1:]:
            p = PolicyGradientModel(observation_space, action_space,
                                    name=env_name,
                                    timestep_limit=T,
                                    num_actors=num_actors,
                                    n_imgfeat=n_imgfeat,
                                    mode="ACKTR",
                                    comb_method=comb_methd,
                                    ent_k=args.ent_k,
                                    session=session,
                                    load_old_model=True,
                                    should_train=False,
                                    parallel_predict=False)
            models.append(p)
        for p in models[1:]:
            p.restore_parameters()
        memory = DictMemory(gamma=args.gamma, lam=args.lam, normalize=True,
                            timestep_limit=T,
                            f_critic=root_model.compute_critic,
                            num_actors=num_actors,
                            f_check_batch=root_model.check_batch_finished,
                            async=True)

        return {"models": models, "memory": memory, "f_should_return_root": f_should_return_root}

    f_create_params = pg_shared_params if len(full_tasks) == 1 else hrl_shared_params

    single_process_mode = True if append_image else False
    experiment = Experiment(f_create_env, f_create_agent,
                            f_create_params, single_process_mode=single_process_mode, render_option="false",
                            log_episodes=True)
    logging.info("run arges: {}".format(args))

    experiment.run_parallelly(num_actors, num_epoch, steps_per_epoch,
                              with_testing_length=test_length)

if __name__ == '__main__':
    main()
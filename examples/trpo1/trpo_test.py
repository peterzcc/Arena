import os

import numpy as np
from arena.games.gym_wrapper import GymWrapper
from arena.games.complex_wrapper import ComplexWrapper
from arena.experiment import Experiment
from arena.agents.test_mp_agent import TestMpAgent
import gym
import argparse
from batch_agent import BatchUpdateAgent
import logging
from custom_ant import CustomAnt
from gather_env import GatherEnv
from maze_env import MazeEnv
from custom_pend import CustomPend
from single_gather_env import SingleGatherEnv
import sys
import os

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

def main():
    parser = argparse.ArgumentParser(description='Script to test the network on cartpole swingup.')
    parser.add_argument('--lr', required=False, default=0.0001, type=float,
                        help='learning rate of the choosen optimizer')
    parser.add_argument('--optimizer', required=False, type=str, default='sgd',
                        help='choice of the optimizer, adam or sgd')
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
    args = parser.parse_args()

    should_profile = False
    if should_profile:
        import yappi

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
    DIRECTIONS = np.array([(1., 0), (0, 1.), (-1., 0), (0, -1.)])

    def random_direction():
        choice = int(4 * np.random.rand())
        return DIRECTIONS[choice, :]
    def f_create_env():
        # env = GatherEnv()
        # env = gym.make('Ant-v1')
        # env = MazeEnv()
        # env = gym.make('InvertedPendulum-v1')

        # return GymWrapper(env,
        #                   max_null_op=0, max_episode_length=T)
        # env = CustomAnt(file_path=cwd + "/cust_ant.xml")
        env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=True,
                              f_gen_obj=random_direction)
        final_env = ComplexWrapper(env, max_episode_length=T,
                                   append_image=True, new_img_size=(64, 64), rgb_to_gray=True,
                                   s_transform=ident,
                                   visible_state_ids=range(env.observation_space.shape[0]),
                                   num_frame=1)
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
        return BatchUpdateAgent(
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid,
            timestep_limit=T
        )

        # return TestMpAgent(
        #     observation_space, action_space,
        #     shared_params, stats_rx, acts_tx,
        #     is_learning, global_t, pid
        # )

    def const_batch_size(n_update):
        return args.batch_size

    def const_target_kl(n_update):
        return 0.003 * args.batch_size / 10000

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

    def f_create_shared_params():
        from multi_trpo_model import MultiTrpoModel
        sample_env = f_create_env()
        observation_space = sample_env.observation_space
        action_space = sample_env.action_space
        sample_env.env.close()
        model = MultiTrpoModel(observation_space, action_space,
                               timestep_limit=T,
                               num_actors=num_actors,
                               f_batch_size=const_batch_size,
                               batch_mode="timestep",
                               f_target_kl=const_target_kl,
                               recompute_old_dist=False,
                               n_imgfeat=None,
                               mode="ADA_KL",
                               update_per_epoch=4,
                               kl_history_length=1)
        return {"global_model": model}

    experiment = Experiment(f_create_env, f_create_agent,
                            f_create_shared_params, single_process_mode=True, render_option="false",
                            log_episodes=True)

    if should_profile:
        yappi.start(builtins=True, profile_threads=True)

    experiment.run_parallel_training(num_actors, num_epoch, steps_per_epoch,
                                     with_testing_length=test_length)
    if should_profile:
        yappi.stop()
        pstat = yappi.convert2pstats(yappi.get_func_stats())
        pstat.dump_stats("profile.out")


if __name__ == '__main__':
    main()
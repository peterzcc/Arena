import os

os.environ["MXNET_GPU_WORKER_NTHREADS"] = "20"
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "32"
import numpy as np
from arena.games.gym_wrapper import GymWrapper
from arena.experiment import Experiment
import gym
import argparse
from trpo_model import TrpoModel
from batch_agent import BatchUpdateAgent
import logging

root = logging.getLogger()
root.setLevel(logging.DEBUG)
import multiprocessing as mp


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
    parser.add_argument('--nactor', required=False, type=int, default=1,
                        help='Number of parallel actor-learners')
    parser.add_argument('--batch-size', required=False, type=int, default=5000,
                        help='batch size')
    parser.add_argument('--num-steps', required=False, type=int, default=25000 * 500,
                        help='Number of parallel actor-learners')
    parser.add_argument('--lr-decrease', default=True, type=bool, help='whether to decrease lr')
    args = parser.parse_args()

    should_profile = False
    if should_profile:
        import yappi

    # Each trajectory will have at most 500 time steps
    T = 1000
    num_actors = args.nactor
    steps_per_epoch = 4000
    num_epoch = int(args.num_steps / steps_per_epoch)
    lr_schedule_interval = 100
    num_updates = int(args.num_steps / (args.batch_size * 100))
    final_factor = 0.01
    lr_factor = final_factor ** (1 / num_updates)

    # if args.gpu < 0:
    #     ctx = mx.cpu()
    # else:
    #     ctx = mx.gpu(args.gpu)

    def f_create_env():
        env = gym.make("InvertedPendulum-v1")
        return GymWrapper(env, max_null_op=0, max_episode_length=T)

    def f_create_agent(observation_space, action_space,
                       shared_params, stats_rx, acts_tx,
                       is_learning, global_t, pid):
        return BatchUpdateAgent(
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid,
            batch_size=args.batch_size
        )

    def f_create_shared_params():
        return None

    experiment = Experiment(f_create_env, f_create_agent,
                            f_create_shared_params, single_process_mode=True)

    test_length = 0
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

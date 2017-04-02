import os

os.environ["MXNET_GPU_WORKER_NTHREADS"] = "20"
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "32"
import numpy as np
from arena.operators import *
from arena.utils import *
from arena import Base
from arena.agents import Agent
from arena.games.gym_wrapper import GymWrapper
from arena.experiment import Experiment
import gym
import argparse
from cont_a3c_agent import ContA3CAgent
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
    parser.add_argument('--batch-size', required=False, type=int, default=4000,
                        help='Number of parallel actor-learners')
    parser.add_argument('--num-steps', required=False, type=int, default=4000 * 500,
                        help='Number of parallel actor-learners')
    parser.add_argument('--lr-decrease', default=True, type=bool, help='whether to decrease lr')
    args = parser.parse_args()

    should_profile = False
    if should_profile:
        import yappi


    # Each trajectory will have at most 500 time steps
    T = 100000
    num_actors = args.nactor
    steps_per_epoch = 4000
    num_epoch = int(args.num_steps / steps_per_epoch)
    lr_schedule_interval = 100
    num_updates = int(args.num_steps / (args.batch_size * 100))
    final_factor = 0.01
    lr_factor = final_factor ** (1 / num_updates)

    if args.gpu < 0:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    def f_create_env():
        env = gym.make("InvertedPendulum-v1")
        return GymWrapper(env, max_null_op=0, max_episode_length=T)

    def f_create_agent(observation_space, action_space,
                       shared_params, stats_rx, acts_tx,
                       is_learning, global_t, pid):
        from mxnet.lr_scheduler import FactorScheduler
        if args.lr_decrease:
            lr_scheduler = FactorScheduler(lr_schedule_interval, lr_factor)
        else:
            lr_scheduler = None
        return ContA3CAgent(
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid,
            ctx=ctx,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_name=args.optimizer,
            lr_scheduler=lr_scheduler
        )

    def f_create_shared_params():
        sample_env = f_create_env()
        sample_agent = f_create_agent(sample_env.observation_space,
                                      sample_env.action_space,
                                      None, None, None,
                                      None, None, -1)
        param_lock = mp.Lock()
        return {"global_net": sample_agent.net,
                "lock": param_lock,
                "updater": sample_agent.updater}

    experiment = Experiment(f_create_env, f_create_agent,
                            f_create_shared_params)

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

import numpy as np
import mxnet as mx
from arena.operators import *
from mxnet.lr_scheduler import FactorScheduler
from arena.utils import *
from arena import Base
from arena.agents import Agent
from arena.games.gym_wrapper import GymWrapper
from arena.experiment import Experiment
# from arena.games.cartpole_box2d import CartpoleSwingupEnv
import gym
import argparse
from cont_a3c_agent import ContA3CAgent
import logging

root = logging.getLogger()
root.setLevel(logging.DEBUG)
import multiprocessing as mp


def main():
    parser = argparse.ArgumentParser(description='Script to test the network on cartpole swingup.')
    parser.add_argument('--lr', required=False, type=float, help='learning rate of the choosen optimizer')
    parser.add_argument('--optimizer', required=False, type=str, default='sgd',
                        help='choice of the optimizer, adam or sgd')
    parser.add_argument('--clip-gradient', default=True, type=bool, help='whether to clip the gradient')
    parser.add_argument('--save-model', default=False, type=bool, help='whether to save the final model')
    parser.add_argument('--gpu', required=False, type=int, default=0,
                        help='Running Context.')
    parser.add_argument('--nactor', required=False, type=int, default=1,
                        help='Number of parallel actor-learners')
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 0.01

    # Each trajectory will have at most 500 time steps
    T = 500
    num_actors = args.nactor

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
        return ContA3CAgent(
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid,
            ctx=ctx,
            batch_size=int(4000 / num_actors),
            lr=args.lr,
            optimizer_name=args.optimizer,
        )

    def f_create_shared_params():
        sample_env = f_create_env()
        sample_agent = ContA3CAgent(
            sample_env.observation_space,
            sample_env.action_space,
            None, None, None,
            None, None, None)
        param_lock = mp.Lock()
        return {"acnet": sample_agent.params,
                "lock": param_lock,
                "updater": sample_agent.updater}

    experiment = Experiment(f_create_env, f_create_agent,
                            f_create_shared_params)

    num_epoch = 500
    steps_per_epoch = 4000 / num_actors
    test_length = 0

    experiment.run_parallel_training(num_actors, num_epoch, steps_per_epoch,
                                     with_testing_length=test_length)


if __name__ == '__main__':
    main()

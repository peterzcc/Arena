import gym
from arena.operators import dqn_sym_nature,DQNInitializer
import arena.agents as agents
import numpy as np
from arena.e_greedy import EpsGreedy
from arena.games.gym_wrapper import GymWrapper
from arena.agents.test_mp_agent import TestMpAgent
from arena.agents.dqn_agent import DqnAgent
from arena.experiment import Experiment
import mxnet as mx
import sys
import argparse
import logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
# root.addHandler(ch)
DEBUG=False

def main():
    parser = argparse.ArgumentParser(description='Script to test the refactored dqn.')
    parser.add_argument('--gpu', required=False, type=int, default=0,
                        help='Running Context.')
    args = parser.parse_args()
    exploration_period = 1000000
    f_get_sym = dqn_sym_nature
    is_double_q = False
    replay_memory_size = 1000000
    train_start = 50000
    history_length = 4
    training_interval = 4
    minibatch_size = 32
    policy = EpsGreedy(eps_0= np.array([1.0]),eps_t=np.array([0.1]) ,t_max=exploration_period,
                       p_assign=np.array([1.0]))
    discount = 0.99
    freeze_interval = 10000
    if args.gpu < 0:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    def f_create_env():
        env = gym.make("BreakoutNoFrameskip-v0")
        # logging.debug("meanings:{}".format(env.get_action_meanings()))
        return GymWrapper(env, rgb_to_gray=True, new_img_size=(84, 84),
                          max_null_op=30, action_mapping=[0, 1, 2, 3],
                          frame_skip=4, max_recent_two_frames=True)

    def f_create_agent(observation_space, action_space,
                       shared_params, stats_rx, acts_tx,
                       is_learning, global_t, pid):
        return DqnAgent(
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid,
            f_get_sym =f_get_sym,
            is_double_q=is_double_q,
            replay_memory_size=replay_memory_size,
            train_start=train_start,
            history_length=history_length,
            training_interval=training_interval,
            minibatch_size=minibatch_size,
            optimizer=mx.optimizer.create(name='adagrad', learning_rate=0.01, eps=0.01),
            policy=EpsGreedy(eps_0=np.array([1.0]), eps_t=np.array([0.1]), t_max=exploration_period,
                             p_assign=np.array([1.0])),
            initializer=None,
            discount=discount,
            freeze_interval=freeze_interval,
            ctx=ctx
        )

    def f_create_shared_params():
        sample_env = f_create_env()
        sample_agent = DqnAgent(
            sample_env.observation_space,
            sample_env.action_space,
            None, None, None,
            None, None, None,
            f_get_sym=f_get_sym,
            is_double_q=is_double_q,
            replay_memory_size=100,
            train_start=train_start,
            history_length=history_length,
            training_interval=training_interval,
            minibatch_size=minibatch_size,
            policy=policy,
            initializer=DQNInitializer(factor_type="in"),
            discount=discount,
            freeze_interval=freeze_interval,
            ctx=ctx
        )
        return {"qnet":sample_agent.params}

    experiment = Experiment(f_create_env, f_create_agent,
                            f_create_shared_params, single_process_mode=DEBUG)
    num_actors = 1
    num_epoch = 200
    steps_per_epoch = 250000
    test_length = 0

    experiment.run_parallel_training(num_actors, num_epoch, steps_per_epoch,
                                     with_testing_length=test_length)

if __name__ == '__main__':
    main()


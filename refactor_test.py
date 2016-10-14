import gym
import arena.agents as agents
from arena.agents.test_mp_agent import TestMpAgent
from arena.experiment import Experiment
import logging
import mxnet as mx
import sys
root = logging.getLogger()
root.setLevel(logging.DEBUG)
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
# root.addHandler(ch)

def main():
    def f_create_env():
        return gym.make("BreakoutDeterministic-v0")

    def f_create_agent(observation_space, action_space,
                       shared_params, stats_rx, acts_tx,
                       is_learning, global_t, pid):
        return TestMpAgent(observation_space, action_space,
                           shared_params, stats_rx, acts_tx,
                           is_learning, global_t, pid)

    def f_create_shared_params():
        return mx.nd.zeros((1024*1024*2),ctx=mx.cpu())


    experiment = Experiment(f_create_env,f_create_agent,
                            f_create_shared_params)

    experiment.run_parallel_training(4,2,10000,with_testing_length=1000)

if __name__ == '__main__':
    main()


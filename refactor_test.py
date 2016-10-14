import gym
import arena.agents as agents
from arena.experiment import Experiment
import logging

root = logging.getLogger()
root.setLevel(logging.DEBUG)
def main():
    env = gym.make("BreakoutDeterministic-v1")
    agent = agents.DqnAgent(observation_space=env.observation_space,
                        action_space=env.action_space)
    experiment = Experiment(env, agent)
    experiment.run_training(2, 10000, with_testing_length=1000)

if __name__ == '__main__':
    main()


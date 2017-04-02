from __future__ import print_function

import sys, gym
from maze_survival import MazeSurvivalEnv

#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#

# env = gym.make('CartPole-v0' if len(sys.argv)<2 else sys.argv[1])
env = MazeSurvivalEnv()
ACTIONS = env.action_space.n
ROLLOUT_TIME = 10000
SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you
# can test what skip is still usable.
ACTION_MAP = {'w': 0, 's': 1, 'd': 2, 'a': 3}


def get_action():
    raw_a = input("Enter your action:\n")
    try:
        a = ACTION_MAP[raw_a]
    except KeyError:
        try:
            a = int(raw_a)
        except ValueError:
            return get_action()
    return a


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    a = 0
    for t in range(ROLLOUT_TIME):
        env.render()
        print("o:\t{}".format(obser))
        if not skip:
            # print("taking action {}".format(human_agent_action))
            # a = human_agent_action
            a = get_action()

            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        print("r:\t{}\td:\t{}".format(r, done))
        if done: break
        if human_wants_restart: break

    print("Game Over")


print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")

while 1:
    rollout(env)

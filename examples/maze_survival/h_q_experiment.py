import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from hierarchical_agent import Option, HierarchicalAgent

from collections import defaultdict

# matplotlib.style.use('ggplot')
from maze_survival import MazeSurvivalEnv
import plotting

env = MazeSurvivalEnv()


def state_collect_key(x):
    return (x[5] - x[1], x[6] - x[2])


def exit_collect_key(state, action):
    if state[0] > 5 or state[1] > 5:
        return True
    if state[0] == 0 and state[1] == 0 and action == 4:
        return True
    return False


def reward_collect_key(state, action):
    if state[0] == 0 and state[1] == 0 and action == 4:
        return 0
    return -10000


def state_goto_room(x):
    return (x[1], x[2])


def exit_leave_room(state, action):
    if action == 4:
        if (abs(state[0]) == 2 and state[1] == 0) or \
                (abs(state[1]) == 2 and state[0] == 0):
            return True
    return False


def reward_enter_north(state, action):
    if action == 4 and state[0] == 0 and state[1] == 2:
        return 1000
    else:
        return -10000


def reward_enter_south(state, action):
    if action == 4 and state[0] == 0 and state[1] == -2:
        return 1000
    else:
        return -10000


def reward_enter_west(state, action):
    if action == 4 and state[0] == -2 and state[1] == 0:
        return 1000
    else:
        return -10000


def reward_enter_east(state, action):
    if action == 4 and state[0] == 2 and state[1] == 0:
        return 1000
    else:
        return -10000


def state_root(x):
    return x  # (x[0], )+x[3:]


raw_actions = [0, 1, 2, 3, 4]


def hierarchical_q_learning(env, num_steps):
    discount_factor = 1.0
    alpha = 0.5
    root_eps_start = 1.0
    root_eps_end = 0.001
    root_eps_length = 4000
    learn_start = 2000

    key_eps_start = 1.0
    key_eps_end = 0.00
    key_eps_length = 1000

    travel_eps_start = 1.0
    travel_eps_end = 0.00
    travel_eps_length = 1000

    option_get_key = Option(state_collect_key, raw_actions, exit_collect_key, reward_collect_key,
                            eps_start=key_eps_start, eps_end=key_eps_end, eps_length=key_eps_length)

    option_enter_north = Option(state_goto_room, raw_actions, exit_leave_room, reward_enter_north,
                                eps_start=travel_eps_start, eps_end=travel_eps_end, eps_length=travel_eps_length)

    option_enter_south = Option(state_goto_room, raw_actions, exit_leave_room, reward_enter_south,
                                eps_start=travel_eps_start, eps_end=travel_eps_end, eps_length=travel_eps_length)

    option_enter_east = Option(state_goto_room, raw_actions, exit_leave_room, reward_enter_east,
                               eps_start=travel_eps_start, eps_end=travel_eps_end, eps_length=travel_eps_length)

    option_enter_west = Option(state_goto_room, raw_actions, exit_leave_room, reward_enter_west,
                               eps_start=travel_eps_start, eps_end=travel_eps_end, eps_length=travel_eps_length)

    primary_options = [option_get_key, option_enter_north, option_enter_south, option_enter_east,
                       option_enter_west] + raw_actions

    root_option = Option(state_root, primary_options, None, None,
                         eps_start=root_eps_start, eps_end=root_eps_end, eps_length=root_eps_length,
                         is_learning=False)
    agent = HierarchicalAgent(root_option)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=[],
        episode_rewards=[])
    # The policy we're following
    # policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    # env.monitor.start('./tmp/maze-experiment-1',force=True)
    i_episode = 0
    i_steps = 0
    while i_steps < num_steps:
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print(
                "\rStep {}/{}.\tepsr:{}\tepsk:{}\tepst:{}\t".format(
                    i_steps + 1, num_steps, root_option.policy.curr_eps,
                    option_get_key.policy.curr_eps,
                    option_enter_east.policy.curr_eps
                ), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        # One step in the environment
        # total_reward = 0.0
        stats.episode_rewards.append(0)
        stats.episode_lengths.append(0)
        for t in itertools.count():
            action = agent.make_decision(state)
            next_state, reward, done, _ = env.step(action)
            agent.receive_feedback(next_state, reward, done)
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            state = next_state
            i_steps += 1

            if done:
                stats.episode_lengths[i_episode - 1] = t
                break
            if i_steps >= num_steps:
                break
        if i_episode == learn_start:
            root_option.is_learning = True
        i_episode += 1
    # env.monitor.close()None
    # gym.upload('./tmp/maze-experiment-1', api_key='sk_CkE85kehTBOS7BDkOYHN2g')
    stats = plotting.EpisodeStats(
        episode_lengths=np.asarray(stats.episode_lengths),
        episode_rewards=np.asarray(stats.episode_rewards))
    stopped = False
    plotting.plot_episode_stats(stats)
    root_option.debug_output = True

    while not stopped:

        key_in = input("Enter a to start:")
        if key_in != 'a':
            continue
        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0

        for t in itertools.count():
            env.render()
            key_in = input("Enter c to stop, any other key to proceed:")
            if key_in == 'c':
                stopped = True
                break
            # Take a step
            action = agent.make_decision(state)
            next_state, reward, done, _ = env.step(action)
            agent.receive_feedback(next_state, reward, done)

            print("r:\t{}\td:\t{}".format(reward, done))
            if done:
                break

            state = next_state


def test_with_flat_option(env, num_steps):
    discount_factor = 1.0
    alpha = 0.5
    root_eps_start = 1.0
    root_eps_end = 0.001
    root_eps_length = 1000

    root_option = Option(lambda x: x, raw_actions, None, None,
                         eps_start=root_eps_start, eps_end=root_eps_end, eps_length=root_eps_length)
    agent = HierarchicalAgent(root_option)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=[],
        episode_rewards=[])
    # The policy we're following
    # policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    # env.monitor.start('./tmp/maze-experiment-1',force=True)
    i_episode = 0
    i_steps = 0
    while i_steps < num_steps:
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rStep {}/{}.".format(i_steps + 1, num_steps), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        # One step in the environment
        # total_reward = 0.0
        stats.episode_rewards.append(0)
        stats.episode_lengths.append(0)
        for t in itertools.count():
            action = agent.make_decision(state)
            next_state, reward, done, _ = env.step(action)
            agent.receive_feedback(next_state, reward, done)
            # Update statistics
            stats.episode_rewards[i_episode] += reward

            if done:
                stats.episode_lengths[i_episode] = t
                break

            state = next_state
            i_steps += 1
        i_episode += 1
    # env.monitor.close()None
    # gym.upload('./tmp/maze-experiment-1', api_key='sk_CkE85kehTBOS7BDkOYHN2g')
    stats = plotting.EpisodeStats(
        episode_lengths=np.asarray(stats.episode_lengths),
        episode_rewards=np.asarray(stats.episode_rewards))
    stopped = False
    plotting.plot_episode_stats(stats)

    while not stopped:
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_steps), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            env.render()
            key_in = input("Enter c to stop, any other key to proceed:\n")
            if key_in == 'c':
                stopped = True
                break
            # Take a step
            action = agent.make_decision(state)
            next_state, reward, done, _ = env.step(action)

            print("r:\t{}\td:\t{}".format(reward, done))
            if done:
                break

            state = next_state


hierarchical_q_learning(env, num_steps=7000000)

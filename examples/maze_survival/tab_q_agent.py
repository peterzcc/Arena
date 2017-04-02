import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

from collections import defaultdict

# matplotlib.style.use('ggplot')
from maze_survival import MazeSurvivalEnv
import plotting

env = MazeSurvivalEnv()


# env = gym.make("Taxi-v1")


def e_greedy_policy(observation, Q, epsilon, nA=5):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)
    return A


def array_to_string(state):
    return ''.join((chr(s + 128) for s in state))


def string_to_array(state_string):
    return np.array((ord(s) - 128 for s in state_string))


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, num_steps, discount_factor=1.0, alpha=0.5, eps_end=0.1,
               eps_length=100):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    eps_start = 1.0
    eps_decay = (eps_start - eps_end) / eps_length

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
        state = array_to_string(env.reset())
        curr_eps = max(eps_start - eps_decay * i_episode, eps_end)
        # One step in the environment
        # total_reward = 0.0
        stats.episode_rewards.append(0)
        stats.episode_lengths.append(0)
        for t in itertools.count():

            # Take a step
            action_probs = e_greedy_policy(state, Q, curr_eps, nA=env.action_space.n)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            next_state = array_to_string(next_state)

            # Update statistics
            stats.episode_rewards[i_episode] += reward

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                stats.episode_lengths[i_episode] = t
                break

            state = next_state
            i_steps += 1
        i_episode += 1
    # env.monitor.close()
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
        state = array_to_string(env.reset())

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            env.render()
            key_in = input("Enter your action:\n")
            if key_in == 'c':
                stopped = True
                break
            # Take a step
            action_probs = e_greedy_policy(state, Q, 0.0, nA=env.action_space.n)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            next_state = array_to_string(next_state)

            print("r:\t{}\td:\t{}".format(reward, done))
            if done:
                break

            state = next_state

    return Q, stats


Q, stats = q_learning(env, num_steps=2000000, eps_end=0.001, eps_length=1000)

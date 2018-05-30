import os

import numpy as np
from arena.games.gym_wrapper import GymWrapper
from arena.games.complex_wrapper import ComplexWrapper
from arena.experiment import Experiment
import gym
from gym.spaces import Box, Discrete
import argparse
from batch_agent import BatchUpdateAgent
from async_agent import AsyncAgent
from hrl_agent import HrlAgent
from flexible_hrl_agent import FlexibleHrlAgent
import logging
from custom_ant import CustomAnt
from gather_env import GatherEnv
from maze_env import MazeEnv
from custom_pend import CustomPend
from single_gather_env import SingleGatherEnv, SimpleSingleGatherEnv
import sys
import os
from collections import OrderedDict
from tf_utils import aggregate_feature, concat_feature, concat_without_task, str2bool
import subprocess
from tensorflow.python.client import device_lib

DIRECTIONS = np.array([(1., 0), (0, 1.), (-1., 0), (0, -1.)])


def x_forward_obj():
    return np.array((1, 0))


def x_backward_obj():
    return np.array((-1, 0))


def x_up_obj():
    return np.array((0, 1))


def x_down_obj():
    return np.array((0, -1))


def v_11():
    return np.sqrt(1 / 2) * np.array((1, 1))


def v_1n1():
    return np.sqrt(1 / 2) * np.array((1, -1))


def v_n11():
    return np.sqrt(1 / 2) * np.array((-1, 1))


def v_n1n1():
    return np.sqrt(1 / 2) * np.array((-1, -1))


def random_direction():
    choice = np.random.randint(0, 4)
    return DIRECTIONS[choice, :]


def up_for():
    choice = np.random.randint(0, 2)
    return DIRECTIONS[choice, :]


def random_cont_direction():
    a = np.random.rand() * 2 * np.pi
    return np.array([np.cos(a), np.sin(a)])


def x_for_back():
    choice = np.random.randint(0, 2) * 2
    return DIRECTIONS[choice, :]


cwd = os.getcwd()


def gen_env_func(env_name, withimg, T=1000):
    append_image = withimg
    feat_sup = False
    task1d = OrderedDict(move0=x_forward_obj, move1=x_backward_obj)
    hrl0 = OrderedDict(move1d=x_for_back, move0=x_forward_obj, move1=x_backward_obj)
    task4 = OrderedDict(move0=x_forward_obj, move1=x_backward_obj,
                        move2=x_up_obj, move3=x_down_obj)
    task8 = OrderedDict(move0=x_forward_obj, move1=x_backward_obj,
                        move2=x_up_obj, move3=x_down_obj,
                        move4=v_11, move5=v_n1n1, move6=v_1n1, move7=v_n11)

    dir_funcs_task8 = list(task8.values())

    def random_direction8():
        choice = np.random.randint(0, len(dir_funcs_task8))
        return dir_funcs_task8[choice]()

    hrl_move2d8 = OrderedDict(move2d8=random_direction8, **task8)

    hrl_move2d = OrderedDict(move2d=random_direction, **task4
                             )
    hrl2 = OrderedDict(reach2d=random_direction,
                       move0=x_forward_obj, move1=x_backward_obj,
                       move2=x_up_obj, move3=x_down_obj
                       )
    hrl_dimage = OrderedDict(moves2d=random_direction,
                             moves0=x_forward_obj, moves1=x_backward_obj,
                             moves2=x_up_obj, moves3=x_down_obj
                             )
    hrl_changing_goal = OrderedDict(dynamic2d=random_direction,
                                    move0=x_forward_obj, move1=x_backward_obj,
                                    move2=x_up_obj, move3=x_down_obj
                                    )
    hrl_dynamic2d5 = OrderedDict(dynamic2d5=random_direction,
                                 **task4)
    hrl_task8train = OrderedDict(task8train=random_direction8,
                                 **task8)
    hrl_dynamic2d5task8 = OrderedDict(dynamic2d5task8=random_direction8,
                                      **task8)
    hrl_c1 = OrderedDict(reachc1=random_direction,
                         move0=x_forward_obj, move1=x_backward_obj,
                         move2=x_up_obj, move3=x_down_obj
                         )
    hrl_c05 = OrderedDict(reachc05=random_direction,
                          move0=x_forward_obj, move1=x_backward_obj,
                          move2=x_up_obj, move3=x_down_obj
                          )
    hrl_fake = OrderedDict(**{"cartpole_hrl": "", "CartPole-v1_0": "", "CartPole-v1_1": ""})
    hrl_up_for = OrderedDict(move_up_for=up_for, move0=x_forward_obj, move2=x_up_obj)
    hrl_simple1d = OrderedDict(simplehrl1d=x_for_back, move0=x_forward_obj, move1=x_backward_obj)

    hrl_root_tasks = dict(move1d=hrl0, move2d=hrl_move2d, reach2d=hrl2, dynamic2d=hrl_changing_goal,
                          reachc1=hrl_c1, reachc05=hrl_c05, moves2d=hrl_dimage, cartpole_hrl=hrl_fake,
                          move_up_for=hrl_up_for, simplehrl1d=hrl_simple1d, move2d8=hrl_move2d8,
                          dynamic2d5=hrl_dynamic2d5, dynamic2d5task8=hrl_dynamic2d5task8, task8train=hrl_task8train)

    full_tasks = [env_name]
    NOISE = "NOISE"
    if env_name in hrl_root_tasks:
        full_tasks = hrl_root_tasks[env_name]

    def f_create_env(render_lock=None, pid=0):
        # env = GatherEnv()
        if env_name == "custant":
            env = CustomAnt(file_path=cwd + "/cust_ant.xml")
        elif env_name in task8:
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=task8[env_name],
                                  reset_goal_prob=0, )
        elif env_name == "forward_and_backward":
            with_state_task = not (append_image and not feat_sup)
            f_direction = x_forward_obj if pid % 2 == 0 else x_backward_obj
            logging.info("actuator[{}], direction: {}".format(pid, f_direction))
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                  f_gen_obj=f_direction)

        elif env_name in hrl0:
            with_state_task = False
            if env_name == list(hrl_root_tasks[env_name].keys())[0]:
                use_internal_reward = False
            else:
                use_internal_reward = True
            subtask_dirs = np.stack([v() for (k, v) in list(hrl0.items())[1:]], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                  f_gen_obj=hrl0[env_name],  # x_forward_obj if pid % 2 else x_backward_obj,#
                                  reset_goal_prob=0,
                                  use_internal_reward=use_internal_reward,
                                  subtask_dirs=subtask_dirs)
        elif env_name == "simplehrl1d":
            with_state_task = False
            if env_name == list(hrl_root_tasks[env_name].keys())[0]:
                use_internal_reward = False
            else:
                use_internal_reward = True
            subtask_dirs = np.stack([v() for (k, v) in list(hrl0.items())[1:]], axis=0)
            env = SimpleSingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=with_state_task,
                                        f_gen_obj=x_for_back,  # x_forward_obj if pid % 2 else x_backward_obj,#
                                        reset_goal_prob=0,
                                        use_internal_reward=use_internal_reward,
                                        subtask_dirs=subtask_dirs)
        elif env_name in hrl_move2d:
            subtask_dirs = np.stack([v() for (k, v) in list(hrl_move2d.items())[1:]], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=hrl_move2d[env_name],
                                  reset_goal_prob=0.0,
                                  subtask_dirs=subtask_dirs)

        elif env_name == "dynamic2d":
            subtask_dirs = np.stack([v() for (k, v) in list(hrl_changing_goal.items())[1:]], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=hrl_changing_goal[env_name],
                                  reset_goal_prob=0.01,
                                  subtask_dirs=subtask_dirs,
                                  use_internal_reward=False)
        elif env_name == "dynamic2d5":
            subtask_dirs = np.stack([v() for (k, v) in list(task4.items())], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=hrl_dynamic2d5[env_name],
                                  reset_goal_prob=0.005,
                                  subtask_dirs=subtask_dirs,
                                  use_internal_reward=False,
                                  constraint_height=False)
        elif env_name == "dynamic2d5task8":
            subtask_dirs = np.stack([v() for (k, v) in list(task8.items())], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_direction8,
                                  reset_goal_prob=0.005,
                                  subtask_dirs=subtask_dirs,
                                  use_internal_reward=False,
                                  constraint_height=False)
        elif env_name == "task8train":
            subtask_dirs = np.stack([v() for (k, v) in list(task8.items())], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=x_forward_obj,
                                  reset_goal_prob=0,
                                  subtask_dirs=subtask_dirs,
                                  use_internal_reward=True,
                                  constraint_height=False)
        elif env_name == "reach_test":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_direction,
                                  use_sparse_reward=True)
        elif env_name == "reach2d":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_direction,
                                  use_sparse_reward=True, )
        elif env_name == "reachc1":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_direction,
                                  catch_range=1,
                                  obj_dist=1.25,
                                  use_sparse_reward=True, )
        elif env_name == "reachc05":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_direction,
                                  catch_range=0.5,
                                  obj_dist=1.25,
                                  use_sparse_reward=True, )
        elif env_name in hrl_dimage:
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=hrl_dimage[env_name],
                                  reset_goal_prob=0, )
        elif env_name == "flatmove1d":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=x_for_back,
                                  forward_scale=1.0,
                                  reset_goal_prob=0, )
        elif env_name == "simplemove1d":
            env = SimpleSingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                        f_gen_obj=x_for_back,
                                        forward_scale=10.0,
                                        reset_goal_prob=0, )
        elif env_name == "scalemove1d":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=x_for_back,
                                  forward_scale=10.0,
                                  reset_goal_prob=0, )
        elif env_name == "scalemove4":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=task8["move4"],
                                  forward_scale=10.0,
                                  reset_goal_prob=0, )
        elif env_name == "cartpole_hrl":
            env = gym.make("CartPole-v1")
        elif env_name in hrl_up_for:
            subtask_dirs = np.stack([v() for (k, v) in list(hrl_root_tasks[env_name].items())[1:]], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=hrl_root_tasks[env_name][env_name],
                                  reset_goal_prob=0, subtask_dirs=subtask_dirs)
        elif env_name == "move2d8":
            subtask_dirs = np.stack([v() for v in list(hrl_move2d8.values())[1:]], axis=0)
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=hrl_move2d8[env_name],
                                  reset_goal_prob=0.0,
                                  subtask_dirs=subtask_dirs)
        elif env_name == "flatcont2d":
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=random_cont_direction,
                                  forward_scale=1.0,
                                  reset_goal_prob=0, )
        elif env_name.startswith(NOISE):
            init_noise = float(env_name[len(NOISE):])
            env = SingleGatherEnv(file_path=cwd + "/cust_ant.xml", with_state_task=False,
                                  f_gen_obj=x_forward_obj,
                                  forward_scale=0.0,
                                  reset_goal_prob=0,
                                  init_noise=init_noise,
                                  constraint_height=False)
        else:
            env = gym.make(env_name)

        dummy_image = False
        final_env = ComplexWrapper(env, max_episode_length=T,
                                   append_image=append_image, rgb_to_gray=True,
                                   visible_state_ids=range(env.observation_space.shape[0]),
                                   num_frame=1,
                                   dummy_image=dummy_image,
                                   render_lock=render_lock)
        # logging.info("created env")
        return final_env

    is_fake_hrl = env_name == list(hrl_fake.keys())[0]
    return f_create_env, full_tasks, is_fake_hrl
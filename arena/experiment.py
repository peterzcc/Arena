import gym
from arena.agents.test_mp_agent import Agent
from arena.actuator import Actuator
from arena.utils import ProcessState, force_map
from time import time
import math
import logging
import os
import multiprocessing as mp
import threading as thd
import mxnet as mx
import numpy as np
import ctypes
import pstats

class Experiment(object):
    def __init__(self,
                 f_create_env,
                 f_create_agent,
                 f_create_shared_params,
                 stats_file_dir=None):
        """

        Parameters
        ----------
        env : gym.Env
        agent : Agent
        """
        self.env = f_create_env()
        self.f_create_env = f_create_env
        self.f_create_agent = f_create_agent
        self.shared_params = f_create_shared_params()
        self.is_learning = mp.Value(ctypes.c_bool, lock=False)
        self.global_t = mp.Value(ctypes.c_int, lock=True)




        if stats_file_dir is None:
            experiment_id = 1
            self.stats_file_dir = "exp_{:d}".format(experiment_id)
            while os.path.exists(self.stats_file_dir):
                experiment_id += 1
                self.stats_file_dir = "exp_{:d}".format(experiment_id)
        else:
            self.stats_file_dir = stats_file_dir
        if not os.path.exists(self.stats_file_dir):
            os.mkdir(self.stats_file_dir)
        logging.info("Saving data at: {}".format(self.stats_file_dir))
        self.log_train_path = os.path.join(self.stats_file_dir, "train_log.csv")
        self.log_test_path = os.path.join(self.stats_file_dir, "test_log.csv")
        self.agent_save_path = os.path.join(self.stats_file_dir, "agent")

    def terminate_all_actuators(self):
        force_map(lambda x: x.put(ProcessState.stop), self.actuator_channels)
        num_running_actor_left = self.num_actor
        is_stopped = np.zeros(self.num_actor)
        while num_running_actor_left > 0:
            rx_msg = self.episode_q.get(block=True)
            try:
                pid = rx_msg["id"]
                if rx_msg["status"] == ProcessState.stop:
                    is_stopped[pid] = 1
                    num_running_actor_left -= 1
            except KeyError:
                pass
        if not np.all(is_stopped):
            raise ValueError("Not all processes have stopped")


    def run_parallel_training(self,num_actor, num_epoch, epoch_length,
                     with_testing_length=0):

        self.actuator_processes = []
        self.actuator_channels = []
        self.agent_threads = []
        self.episode_q = mp.Queue()
        self.num_actor = num_actor

        def actuator_thread(func_get_env, stats_tx: mp.Queue, acts_rx: mp.Queue,
                            cmd_signal: mp.Queue, episode_data_q: mp.Queue,
                            global_t, act_id=0):
            this_actuator = Actuator(func_get_env, stats_tx, acts_rx,
                                     cmd_signal, episode_data_q,
                                     global_t, act_id)
            this_actuator.run_loop()

        def agent_thread(observation_space, action_space,
                         shared_params, stats_rx: mp.Queue, acts_tx: mp.Queue,
                         is_learning, global_t, pid):

            this_agent = self.f_create_agent(observation_space, action_space,
                                             shared_params, stats_rx, acts_tx,
                                             is_learning, global_t, pid)
            this_agent.run_loop()

        for process_id in range(num_actor):
            self.actuator_channels.append(mp.Queue())
            this_stats_queue = mp.Queue()
            this_action_queue = mp.Queue()
            this_actuator_process = \
                mp.Process(
                    target=actuator_thread,
                    args=(self.f_create_env,
                          this_stats_queue,
                          this_action_queue,
                          self.actuator_channels[process_id],
                          self.episode_q,
                          self.global_t,
                          process_id))
            this_actuator_process.daemon = True
            self.actuator_processes.append(this_actuator_process)
            this_agent_thread = \
                thd.Thread(
                    target=agent_thread,
                    args=(self.env.observation_space,
                          self.env.action_space,
                          self.shared_params,
                          this_stats_queue,
                          this_action_queue,
                          self.is_learning,
                          self.global_t,
                          process_id)
                )
            this_agent_thread.daemon = True
            self.agent_threads.append(this_agent_thread)
        # for actuator in self.actuator_processes:
        #     actuator.start()
        # for agent in self.agent_threads:
        #     agent.start()
        force_map(lambda x: x.start(), self.actuator_processes)
        force_map(lambda x: x.start(), self.agent_threads)

        epoch_num = 0
        epoch_reward = 0
        num_episode = 0
        force_map(lambda x: x.put(ProcessState.start), self.actuator_channels)
        start_times = np.repeat(time(), num_actor)

        if not os.path.exists(self.log_train_path):
            log_train_file = open(self.log_train_path, 'w')
            log_train_file.write(
                "Epoch,t,Episode duration,Reward,fps\n")

            log_train_file.close()

        while epoch_num < num_epoch:
            rx_msg = self.episode_q.get(block=True)
            try:
                pid = rx_msg["id"]
                episode_count = rx_msg["episode_count"]
                episode_reward = rx_msg["episode_reward"]
            except KeyError:
                raise ValueError("Unexpected value of episode queue")

            num_episode += 1
            epoch_reward += episode_reward
            with self.global_t.get_lock():
                self.global_t.value += episode_count

            current_time = time()
            fps = episode_count / (current_time - start_times[pid])
            start_times[pid] = current_time

            with open(self.log_train_path, 'a') as log_train_file:
                train_log = ",".join(
                    map(str,
                        [epoch_num, self.global_t.value, episode_count, episode_reward, round(fps)]
                        )) + "\n"
                log_train_file.write(train_log)

            if self.global_t.value > (epoch_num+1)*epoch_length:
                if with_testing_length > 0:
                    self.terminate_all_actuators()
                    self.is_learning.value = False
                    self.run_testing_on_sub_process(with_testing_length)
                    force_map(lambda x: x.put(ProcessState.start), self.actuator_channels)
                    start_times = np.repeat(time(), num_actor)

                epoch_num += 1


    def run_testing_on_sub_process(self,test_length,process_id = 0):
        if not os.path.exists(self.log_test_path):
            with open(self.log_test_path, 'w') as log_test_file:
                log_test_file.write("id,mean reward, episode_num, fps\n")
        self.actuator_channels[process_id].put(ProcessState.start)
        test_t = 0
        test_reward = 0
        test_episode_num = 0
        start_time = time()
        while test_t < test_length:
            rx_msg = self.episode_q.get(block=True)
            try:
                if rx_msg["id"] != process_id:
                    raise ValueError("Process id is not correct")
                episode_reward = rx_msg["episode_reward"]
                episode_count = rx_msg["episode_count"]
            except KeyError:
                raise ValueError("message error during testing")
            test_reward += episode_reward
            test_episode_num += 1
            test_t += episode_count
        end_time = time()
        self.actuator_channels[process_id].put(ProcessState.stop)
        process_stopped = False
        while not process_stopped:
            rx_msg = self.episode_q.get(block=True)
            if "status" in rx_msg:
                if rx_msg["status"] == ProcessState.stop:
                    process_stopped = True
        with open(self.log_test_path, 'a') as log_test_file:
            log_test_file.write(",".join(map(str, [process_id,
                                                   test_reward/test_episode_num,
                                                   test_episode_num,
                                                   round(test_t/(end_time-start_time))]))+"\n")






    def run_training(self, num_epoch, epoch_length, max_episode_length=math.inf,
                     with_testing_length=0):
        """

        Parameters
        ----------
        num_epoch
        epoch_length
        max_episode_length

        Returns
        -------

        """
        if not os.path.exists(self.log_train_path):
            log_train_file = open(self.log_train_path, 'w')
            log_train_file.write(
                "Epoch,Episode,Episode duration,Reward,fps,{}\n".format(
                    ",".join(map(str, self.agent.stats_keys()))
                ))
            log_train_file.close()
        total_steps = 0

        for epoch_num in range(num_epoch):
            log_train_file = open(self.log_train_path, 'a')
            steps_left = epoch_length
            episode_num = 0
            epoch_reward = 0

            while steps_left > 0:
                episode_num += 1
                episode_num_step = 0
                episode_reward = 0

                episode_ends = False
                first_obs = self.env.reset()
                observation = first_obs
                reward = 0
                epso_start_time = time()

                while not episode_ends:
                    this_action = self.agent.act(observation)
                    observation, reward, episode_ends, info_env = self.env.step(this_action)
                    self.agent.receive_feedback(reward, episode_ends)
                    episode_reward += reward
                    total_steps += 1
                    episode_num_step += 1
                    # handle max length
                    episode_ends = episode_ends or (episode_num_step >= max_episode_length)
                steps_left -= episode_num_step
                fps = episode_num_step / (time() - epso_start_time)
                train_log = ",".join(
                        map(str,
                            [epoch_num, episode_num, episode_num_step, episode_reward, round(fps)] + self.agent.stats_values()
                            ))+"\n"
                log_train_file.write(train_log)

                epoch_reward += episode_reward
                episode_num += 1

            logging.info("training epoch: {}, reward: {}".format(epoch_num, epoch_reward/episode_num))
            self.agent.save_parameters(self.agent_save_path)
            if with_testing_length > 0:
                self.run_testing(with_testing_length, str(epoch_num),
                                 max_episode_length=max_episode_length)
            epoch_num += 1
            log_train_file.close()

    def run_testing(self,test_length, agent_id="", max_episode_length=math.inf):
        if not os.path.exists(self.log_test_path):
            with open(self.log_test_path, 'w') as log_test_file:
                log_test_file.write("id,mean reward, episode_num\n")
        steps_left = test_length
        episode_num = 0
        epoch_reward = 0

        while steps_left > 0:
            episode_num += 1
            episode_num_step = 0
            episode_reward = 0

            episode_ends = False
            first_obs = self.env.reset()
            observation = first_obs
            reward = 0

            while not episode_ends:
                this_action = self.agent.act(observation)
                observation, reward, episode_ends, info_env = self.env.step(this_action)
                episode_reward += reward
                episode_num_step += 1
                # handle max length
                episode_ends = episode_ends or (episode_num_step >= max_episode_length)
            steps_left -= episode_num_step
            epoch_reward += episode_reward
            episode_num += 1

        with open(self.log_test_path, 'a') as log_test_file:
            log_test_file.write(",".join(map(str, [agent_id, epoch_reward/episode_num, episode_num]))+"\n")

    def demo(self):
        episode_ends = True
        while True:
            self.env.render()
            if episode_ends:
                observation = self.env.reset()
            this_action = self.agent.act(observation)
            observation, reward, episode_ends, info_env = self.env.step(this_action)

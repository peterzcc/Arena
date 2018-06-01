import multiprocessing as mp
import gym
# from arena.agents.test_mp_agent import Agent
from arena.actuator import Actuator
from arena.mp_utils import ProcessState, force_map, FastPipe, RenderOption, MultiFastPipe, MpCtxManager
from time import time
import logging
import os
import threading as thd
import numpy as np
import ctypes
import datetime
import queue
import threading
from gym.spaces import Discrete, Box
import signal
import sys
from multiprocessing import process
from arena.games.cust_control import make_env

def space_to_np(space):
    if isinstance(space, Discrete):
        return 0
    elif isinstance(space, Box):
        return space.low
    else:
        raise ValueError("Unsupported arg")


def actuator_thread(env_args, stats_tx, acts_rx,
                    cmd_signal, episode_data_q,
                    global_t, act_id=0, render_option=None):
    # trick to reduce gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    with tf.device('/cpu:0'):
        this_actuator = Actuator(env_args, stats_tx, acts_rx,
                                 cmd_signal, episode_data_q,
                                 global_t, act_id,
                                 render_option=render_option
                                 )
        this_actuator.run_loop()

class Experiment(object):
    """Class for automatically running parallel AI agents


    """
    def __init__(self,
                 env_args,
                 f_create_agent,
                 f_create_shared_params,
                 stats_file_dir=None,
                 single_process_mode=False,
                 render_option="off",
                 log_episodes=False):
        """
        Parameters
        ----------
        env : gym.Env
        agent : Agent
        """
        # 2. Configure log files
        if stats_file_dir is None:
            experiment_id = 1
            self.stats_file_dir = "exp_{:d}".format(experiment_id)
            mkdir_success = False
            while not mkdir_success:
                if os.path.exists(self.stats_file_dir):
                    experiment_id += 1
                    self.stats_file_dir = "exp_{:d}".format(experiment_id)
                else:
                    try:
                        os.mkdir(self.stats_file_dir)
                        mkdir_success = True
                    except FileExistsError:
                        experiment_id += 1
                        self.stats_file_dir = "exp_{:d}".format(experiment_id)
        else:
            self.stats_file_dir = stats_file_dir

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        Experiment.EXP_NAME = self.stats_file_dir

        # create file handler which logs even debug messages
        fh = logging.FileHandler(self.stats_file_dir + '/log.txt', mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        root.addHandler(fh)
        # root.addHandler(ch)

        self.single_process_mode = single_process_mode

        # 1. Store variables
        env, env_info = make_env(**env_args)
        mp_ctx = MpCtxManager.get_mp_ctx()
        if single_process_mode:
            self.process_type = thd.Thread
            self.queue_type = queue.Queue
        else:
            self.process_type = mp_ctx.Process
            self.queue_type = mp_ctx.Queue
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.info_sample = {}
        try:
            self.info_sample = env.info_sample
        finally:
            pass
        self.env_args = env_args
        self.f_create_agent = f_create_agent
        self.f_create_shared_params = f_create_shared_params
        self.shared_params = None  # f_create_shared_params()
        self.is_learning = mp_ctx.Value(ctypes.c_bool, lock=False)
        self.is_learning.value = True
        self.global_t = mp_ctx.Value(ctypes.c_int, lock=True)
        self.global_t.value = 0
        self.actuator_processes = []
        self.actuator_channels = []
        self.agent_threads = []
        self.episode_q = self.queue_type(maxsize=1000)
        self.num_actor = None
        self.render_option = render_option
        self.log_episodes = log_episodes

        print("Saving data at: {}".format(self.stats_file_dir))
        self.log_train_path = os.path.join(self.stats_file_dir, "train_log.csv")
        self.log_test_path = os.path.join(self.stats_file_dir, "test_log.csv")
        Experiment.stats_path = os.path.join(self.stats_file_dir, "stats.h5")

        signal.signal(signal.SIGINT, self.interrupt)
        self.dead_locked = False

    EXP_NAME = "exp_unknown"
    is_terminated = False
    f_terminate = []
    stats_path = None

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

    def create_actor_learner_processes(self, num_actor):
        agents = []

        def agent_run_thread(agent, pid):
            agent.run_loop()


        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")
        process.current_process()._config['tempdir'] = "./tmp"

        if isinstance(self.observation_space, list):
            observation_sample = list(map(space_to_np, self.observation_space))
            obs_type = MultiFastPipe
        else:
            observation_sample = space_to_np(self.observation_space)
            obs_type = FastPipe
        action_sample = space_to_np(self.action_space)
        stats_pipes = []
        action_pipes = []

        for process_id in range(num_actor):
            self.actuator_channels.append(self.queue_type())
            obs_pipe = obs_type({"observation": observation_sample})
            action_pipe = FastPipe({"action": action_sample})
            action_pipes.append(action_pipe)
            feedback_sample = {"reward": 0.0, "done": False}
            feedback_sample.update(self.info_sample)
            feedback_pipe = FastPipe(feedback_sample)
            stats_pipe = [obs_pipe, feedback_pipe]
            stats_pipes.append(stats_pipe)

            this_actuator_process = \
                self.process_type(
                    target=actuator_thread,
                    args=(self.env_args,
                          stats_pipe,
                          action_pipe,
                          self.actuator_channels[process_id],
                          self.episode_q,
                          self.global_t,
                          process_id,
                          RenderOption.lookup(self.render_option)))
            this_actuator_process.daemon = True
            self.actuator_processes.append(this_actuator_process)
        for actuator in self.actuator_processes:
            actuator.start()
        logging.info("actuators started")
        self.shared_params = self.f_create_shared_params()
        for process_id in range(num_actor):
            stats_pipe = stats_pipes[process_id]
            action_pipe = action_pipes[process_id]
            agent = self.f_create_agent(self.observation_space,
                                        self.action_space,
                                        self.shared_params,
                                        stats_pipe,
                                        action_pipe,
                                        self.is_learning,
                                        self.global_t,
                                        process_id)
            agents.append(agent)

        for process_id in range(num_actor):
            this_agent_thread = \
                thd.Thread(
                    target=agent_run_thread,
                    args=(agents[process_id], process_id)
                )
            this_agent_thread.daemon = True
            self.agent_threads.append(this_agent_thread)
        for agent in self.agent_threads:
            agent.start()

    def interrupt(self, signal, frame):
        force_map(lambda x: x.put(ProcessState.terminate), self.actuator_channels)
        print('You pressed Ctrl+C. Terminating..')

        Experiment.is_terminated = True
        self.terminate()
        sys.exit()

    def terminate(self):
        logging.warning("Experiment terminating")
        force_map(lambda x: x.put(ProcessState.terminate), self.actuator_channels)
        # if self.single_process_mode:
        #     for actuator in self.actuator_processes:
        #         actuator.join()
        #     logging.warning("actuators terminated")
        # for agent in self.agent_threads:
        #     agent.join()
        # logging.warning("agents terminated")
        for f in Experiment.f_terminate:
            f()
        # for t in threading.enumerate():
        #     logging.warning('ACTIVE: %s', t.getName())
        logging.warning("finished cleaning")
        sys.exit()

    def run_parallelly(self, num_actor, num_epoch, epoch_length,
                       with_testing_length=0):

        self.num_actor = num_actor
        # Where should we put the creation of actor/learners?

        self.create_actor_learner_processes(num_actor)

        epoch_num = 0
        epoch_reward = 0
        num_episode = 0

        if not os.path.exists(self.log_train_path):
            log_train_file = open(self.log_train_path, 'w')
            log_train_file.write(
                "Reward,pid,t,Episode duration\n")
            log_train_file.close()

        start_times = np.repeat(time(), num_actor)
        force_map(lambda x: x.put(ProcessState.start), self.actuator_channels)

        while epoch_num < num_epoch and not Experiment.is_terminated:
            try:
                rx_msg = self.episode_q.get(block=True, timeout=1)
                self.dead_locked = False
            except queue.Empty:
                if Experiment.is_terminated:
                    self.terminate()
                    return
                # if not self.dead_locked:
                #     logging.warning("Not received message for too long. Maybe there is something wrong")
                #     self.dead_locked = True

                # for (pid, p_actuator) in enumerate(self.actuator_processes):
                #     logging.debug("Actuator {} alive:{}".format(pid, p_actuator.is_alive()))
                # for (pid, agent_thread) in enumerate(self.agent_threads):
                #     logging.debug("Agent {} alive:{}".format(pid, agent_thread.is_alive()))
                continue
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

            # current_time = time()
            # fps = episode_count / (current_time - start_times[pid])
            # start_times[pid] = current_time
            # fps = 0
            if self.log_episodes:
                with open(self.log_train_path, 'a') as log_train_file:
                    train_log = ",".join(
                        map(str,
                            [episode_reward, pid, self.global_t.value, episode_count,
                             ]
                            )) + "\n"
                    log_train_file.write(train_log)

                    if self.global_t.value > (epoch_num + 1) * epoch_length:
                        if with_testing_length > 0:
                            logging.error("testing not implemented")
                            # self.terminate_all_actuators()
                            # self.is_learning.value = False
                            # self.run_testing_on_sub_process(with_testing_length)
                            # self.is_learning.value = True
                            # force_map(lambda x: x.put(ProcessState.start), self.actuator_channels)
                        epoch_num += 1
                # logging.debug("exp: Epoch {} Finished.\n".format(epoch_num))
        logging.info("training finished")
        self.terminate()

    # def run_testing_on_sub_process(self, test_length, process_id=0):
    #     if not os.path.exists(self.log_test_path):
    #         with open(self.log_test_path, 'w') as log_test_file:
    #             log_test_file.write("t,mean reward, episode_num, fps\n")
    #     self.actuator_channels[process_id].put(ProcessState.start)
    #     test_t = 0
    #     test_reward = 0
    #     test_episode_num = 0
    #     start_time = time()
    #     while test_t < test_length:
    #         raise ValueError("not implemented")
    #         rx_msg = self.episode_q.get(block=True)
    #         try:
    #             if rx_msg["id"] != process_id:
    #                 raise ValueError("Process id is not correct")
    #             episode_reward = rx_msg["episode_reward"]
    #             episode_count = rx_msg["episode_count"]
    #         except KeyError:
    #             raise ValueError("message error during testing")
    #         test_reward += episode_reward
    #         test_episode_num += 1
    #         test_t += episode_count
    #     end_time = time()
    #     self.actuator_channels[process_id].put(ProcessState.stop)
    #     process_stopped = False
    #     while not process_stopped:
    #         rx_msg = self.episode_q.get(block=True)
    #         if "status" in rx_msg:
    #             if rx_msg["status"] == ProcessState.stop:
    #                 process_stopped = True
    #     with open(self.log_test_path, 'a') as log_test_file:
    #         log_test_file.write(",".join(map(str, [self.global_t.value,
    #                                                test_reward / test_episode_num,
    #                                                test_episode_num,
    #                                                round(test_t/(end_time-start_time))])) +"\n")

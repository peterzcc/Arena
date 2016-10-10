import gym
from .agents.agent import Agent
from time import time
import math
import logging
import os


class Experiment(object):
    def __init__(self, env, agent, stats_file_dir=None):
        """

        Parameters
        ----------
        env : gym.Env
        agent : Agent
        """
        self.env = env
        self.agent = agent

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
                    this_action = self.agent.act(observation,
                                                 is_learning=True)
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
                            [epoch_num, episode_num,episode_num_step, episode_reward, fps] + self.agent.stats_values()
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

    def run_testing(self,test_length,agent_id="", max_episode_length=math.inf):
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
                this_action = self.agent.act(observation,
                                             is_learning=False)
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
        reward = 0
        episode_ends = True
        while True:
            self.env.render()
            if episode_ends:
                observation = self.env.reset()
            this_action = self.agent.act(observation,
                                         is_learning=False)
            observation, reward, episode_ends, info_env = self.env.step(this_action)

import gym
from gym.spaces import Box, Discrete
import numpy as np
import cv2
import logging

#TODO: test this class
class GymWrapper(object):
    def __init__(self, env: gym.Env, rgb_to_gray=False, new_img_size=None,
                 max_null_op=7, action_mapping=None, frame_skip=1,
                 max_recent_two_frames=False,
                 max_episode_length=100000):
        self.env = env
        if action_mapping is None:
            self.action_space = env.action_space
            self.action_map = None
        else:
            assert isinstance(env.action_space, Discrete)
            self.action_space = Discrete(len(action_mapping))
            self.action_map = action_mapping

        obs_min =np.ravel(self.env.observation_space.low)[0]
        obs_max = np.ravel(self.env.observation_space.high)[0]
        if rgb_to_gray or new_img_size is not None:
            assert len(self.env.observation_space.shape) == 3
            assert self.env.observation_space.shape[2] == 3
            if rgb_to_gray:
                num_channel = 1
            else:
                num_channel = 3
            if new_img_size is None:
                image_size = self.env.observation_space.low.shape[0:2]
            else:
                image_size = new_img_size
            if rgb_to_gray:
                self.observation_space = Box(low=obs_min,
                                             high=obs_max,
                                             shape=image_size
                                             )
            else:
                self.observation_space = Box(low=obs_min,
                                             high=obs_max,
                                             shape=image_size + (num_channel,)
                                             )
        self.observation_space = env.observation_space
        self.rgb_to_gray = rgb_to_gray
        self.new_img_size = new_img_size
        self.max_null_op = max_null_op
        self.screen_buffer_length = 2
        self.frame_skip = frame_skip
        self.max_recent_two_frames = max_recent_two_frames
        self.max_episode_length = max_episode_length

        # Episode information
        self.episode_steps = 0

    def render(self):
        self.env.render()

    def preprocess_observation(self, obs):
        final = obs
        if self.new_img_size is not None:
            final = cv2.resize(obs, self.new_img_size,
                               interpolation=cv2.INTER_LINEAR)
        if self.rgb_to_gray:
            final = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
        return final

    def step(self, a):
        # logging.debug("rx a:{}".format(a))
        final_done = False
        final_reward = 0
        observations = []

        for t_skip in range(self.frame_skip):
            if self.action_map is not None:
                observation, reward, done, info = self.env.step(self.action_map[a])
            else:
                observation, reward, done, info = self.env.step(a)
            observations.append(observation)
            final_done = final_done or done
            final_reward += reward
            if done:
                break
        if self.max_recent_two_frames and len(observations) >= 2:
            final_observation = np.maximum(
                self.preprocess_observation(observations[-1]),
                self.preprocess_observation(observations[-2])
            )
        else:
            final_observation = self.preprocess_observation(observations[-1])

        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_length:
            final_done = True

        # logging.debug("tx r:{},d:{}".format(reward, done))

        # if done:
        #     logging.debug("a:{},r:{},d:{}".format(a, reward, done))
        # else:
        #     logging.debug("a:{},r:{}".format(a, reward))

        return final_observation, final_reward, final_done, info

    def reset(self):
        observation = self.env.reset()
        null_op_num = np.random.randint(
            0,
            max(self.max_null_op + 1, 0 + 1))
        # logging.debug("null_op:{}".format(null_op_num))
        for i in range(null_op_num):
            observation, _, _, _ = self.env.step(0)
        self.episode_steps = 0
        return self.preprocess_observation(observation)


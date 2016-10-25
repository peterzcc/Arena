import gym
from gym.spaces import Box
import numpy as np
import cv2
import logging

#TODO: test this class
class GymWrapper(object):
    def __init__(self, env: gym.Env, rgb_to_gray=True, new_img_size=None,
                 max_null_op=7):
        self.env = env
        self.action_space = env.action_space
        assert len(self.env.observation_space.shape) == 3
        assert self.env.observation_space.shape[2] == 3
        obs_min =np.ravel(self.env.observation_space.low)[0]
        obs_max = np.ravel(self.env.observation_space.high)[0]
        if rgb_to_gray:
            num_channel = 1
        else:
            num_channel = 3
        if new_img_size is None:
            image_size = self.env.observation_space.low.shape[0:2]
        else:
            image_size = new_img_size
        if rgb_to_gray:
            self.observation_space = Box(low=0,
                                         high=255,
                                         shape=image_size
                                         )
        else:
            self.observation_space = Box(low=0,
                                         high=255,
                                         shape=image_size + (num_channel,)
                                         )
        self.rgb_to_gray = rgb_to_gray
        self.new_img_size = new_img_size
        self.max_null_op = max_null_op

    def render(self):
        self.env.render()

    def preprocess_observation(self, obs):

        if self.new_img_size is not None:
            resized = cv2.resize(obs, self.new_img_size,
                                 interpolation=cv2.INTER_LINEAR)
        else:
            resized = obs
        if self.rgb_to_gray:
            final = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        else:
            final = resized

        return final

    def step(self, a):
        # logging.debug("rx a:{}".format(a))
        observation, reward, done, info = self.env.step(a)
        # logging.debug("tx r:{},d:{}".format(reward, done))
        final_observation = self.preprocess_observation(observation)
        # if done:
        #     logging.debug("a:{},r:{},d:{}".format(a, reward, done))
        # else:
        #     logging.debug("a:{},r:{}".format(a, reward))

        return final_observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        null_op_num = np.random.randint(
            0,
            max(self.max_null_op + 1, 0 + 1))
        # logging.debug("null_op:{}".format(null_op_num))
        for i in range(null_op_num):
            observation, _, _, _ = self.env.step(0)
        return self.preprocess_observation(observation)


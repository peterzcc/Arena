import gym
import numpy as np
import cv2
#TODO: test this class
class GymWrapper(object):
    def __init__(self, env: gym.Env, rgb_to_gray=True, new_img_size=None,
                 max_null_op=7):
        self.env = env
        self.action_space = env.action_space

        assert self.env.observation_space.shape[3] == 3
        assert len(self.env.observation_space.shape) == 3
        obs_min = self.env.observation_space.low.ravels()[0]
        obs_max = self.env.observation_space.high.ravels()[0]
        if rgb_to_gray:
            num_channel = 1
        else:
            num_channel = 3
        if new_img_size is None:
            image_size = self.env.observation_space.low.shape[0:2]
        else:
            image_size = new_img_size
        self.observation_space = image_size + (num_channel,)
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
        observation, reward, done, info = self.env.step(a)
        observation = self.preprocess_observation(observation)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        null_op_num = np.random.randint(
            0,
            max(self.max_null_op + 1, 0 + 1))
        for i in range(null_op_num):
            observation, _, _, _ = self.env.step(0) #TODO: define null action
        return observation


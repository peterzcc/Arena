import gym
import numpy as np
#TODO: Complete this class later
class GymWrapper(object):
    def __init__(self, env: gym.Env, rgb_to_gray=True, new_img_size=None,
                 max_null_op=30):
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
        self.observation_space = new_img_size + (num_channel,)
        self.rgb_to_gray = rgb_to_gray
        self.new_img_size = new_img_size
        self.max_null_op=30

    def render(self):
        self.env.render()

    def preprocess_observation(self,obs):
        #TODO: preprocess
        return obs

    def step(self, a):
        observation, reward, done, info = self.env.step(a)
        observation = self.preprocess_observation(observation)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()

        #TODO: null op
        raise NotImplementedError
        return observation


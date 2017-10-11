import gym
from gym.spaces import Box, Discrete
import numpy as np
import cv2
import logging

import time

# TODO: test this class


class ComplexWrapper(object):
    def __init__(self, env: gym.Env, rgb_to_gray=False, new_img_size=None,
                 max_episode_length=100000, action_reduce=False,
                 append_image=False, visible_state_ids=None,
                 s_transform=lambda x, t: x, num_frame=1):
        args = locals()
        logging.debug("Environment args:\n {}".format(args))
        self.env = env
        self.action_reduce = action_reduce
        self.env = env
        self.action_space = env.action_space
        if self.action_reduce:
            self.action_space.low = np.array((self.action_space.low[0],))
            self.action_space.high = np.array((self.action_space.high[0],))
        self.action_map = None
        # self.obs_start = remove_obs_until + 1
        self.vs_id = visible_state_ids
        # obs_min =np.ravel(self.env.observation_space.low)
        # obs_max = np.ravel(self.env.observation_space.high)
        # state_shape = self.env.observation_space.shape
        self.state_space = Box(low=env.observation_space.low[self.vs_id],
                               high=env.observation_space.high[self.vs_id])
        self.observation_space = [self.state_space]
        self.append_image = append_image
        self.s_transform = s_transform
        self.num_frame = num_frame
        if append_image:
            sample_image = self.render(mode="rgb_array")
            image_shape = sample_image.shape
            logging.info("original size: {}".format(image_shape))
            img_min = 0
            img_max = 255
            if rgb_to_gray or new_img_size is not None:
                assert len(image_shape) == 3
                assert image_shape[2] == 3
                if rgb_to_gray:
                    num_channel = 1 * num_frame
                else:
                    num_channel = 3
                if new_img_size is None:
                    image_size = image_shape[0:2]
                else:
                    image_size = new_img_size
                if rgb_to_gray:
                    self.img_space = Box(low=img_min,
                                         high=img_max,
                                         shape=image_size + (num_channel,)
                                         )
                else:
                    self.img_space = Box(low=img_min,
                                         high=img_max,
                                         shape=image_size + (num_channel,)
                                         )
            self.observation_space += [self.img_space]
            self.rgb_to_gray = rgb_to_gray
            self.new_img_size = new_img_size
            self.frame_buffer = self.img_space.low.copy()
            self.x_buffer = self.num_frame - 1
        # else:
        #     self.new_img_size = (0, 0)
        #     self.img_space = Box(low=np.empty(shape=(0, 0, 0)),
        #                          high=np.empty(shape=(0, 0, 0)),
        #                          shape=(0, 0, 0)
        #                          )
        #     self.observation_space += [self.img_space]
        self.max_episode_length = max_episode_length
        self.info_sample = {"terminated": False}




        # Episode information
        self.episode_steps = 0
        self.total_steps = 0

    def render(self, mode='human', close=False):
        # self.render_lock.acquire()
        result = self.env.render(mode=mode, close=close)
        # self.render_lock.release()
        return result

    def preprocess_observation(self, obs):
        final = obs
        if self.new_img_size is not None:
            final = cv2.resize(obs, self.new_img_size,
                               interpolation=cv2.INTER_LINEAR)
        if self.rgb_to_gray:
            final = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
        return final

    def env_step(self, a):
        state_observation, reward, done, info = self.env.step(a)
        state_observation = self.s_transform(state_observation,self.total_steps)
        if self.append_image:
            image_observation = self.render(mode="rgb_array")
            image_observation = self.preprocess_observation(image_observation)
            self.x_buffer = (self.x_buffer + 1) % self.num_frame
            self.frame_buffer[:, :, self.x_buffer] = image_observation
            stacked_obs = np.take(self.frame_buffer, np.arange(self.x_buffer + 1 - self.num_frame, self.x_buffer + 1),
                                  axis=2,
                                  mode='wrap')
            return [state_observation[self.vs_id], stacked_obs], reward, done, info
        else:
            return [state_observation[self.vs_id]], reward, done, info

    def env_reset(self):
        state_observation = self.env.reset()
        state_observation = self.s_transform(state_observation,self.total_steps)
        if self.append_image:
            self.frame_buffer = self.img_space.low.copy()
            self.x_buffer = self.num_frame - 1
            image_observation = self.render(mode="rgb_array")
            image_observation = self.preprocess_observation(image_observation)
            self.frame_buffer[:, :, self.x_buffer] = image_observation
            return [state_observation[self.vs_id], self.frame_buffer.copy()]
        else:
            return [state_observation[self.vs_id]]

    def step(self, a):
        a = np.append(a, (0,))
        # logging.debug("rx a:{}".format(a))
        final_done = False
        final_reward = 0
        info_terminated = {"terminated": False}

        final_observation, reward, done, info = self.env_step(a)
        final_done = final_done or done
        final_reward += reward
        if done:
            info_terminated.update({"terminated": True})

        self.episode_steps += 1


        if self.episode_steps >= self.max_episode_length:
            final_done = True

        return final_observation, final_reward, final_done, info_terminated

    def close(self):
        self.env.close()

    def reset(self):
        observation = self.env_reset()
        self.total_steps += self.episode_steps
        self.episode_steps = 0
        return observation

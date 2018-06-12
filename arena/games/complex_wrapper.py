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
                 s_transform=lambda x, t: x, num_frame=1,
                 dummy_image=False,
                 render_lock=None):
        # args = locals()
        # logging.debug("Environment args:\n {}".format(args))
        self.env = env
        self.metadata = env.metadata
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
        self.dummy_imagge = dummy_image
        self.s_transform = s_transform
        self.num_frame = num_frame
        self.render_lock = render_lock
        if append_image:
            sample_image = self.render(mode="rgb_array")
            image_shape = sample_image.shape
            # logging.info("original size: {}".format(image_shape))
            img_min = 0
            img_max = 255
            assert len(image_shape) == 3
            if image_shape[2] == 1:
                rgb_to_gray = False
            if rgb_to_gray:
                num_channel = 1 * num_frame
            else:
                num_channel = sample_image.shape[2]
            if new_img_size is None:
                image_size = image_shape[0:2]
            else:
                image_size = new_img_size
            final_img_shape = image_size + (num_channel,)
            zero_img = np.zeros(final_img_shape, np.uint8)
            if rgb_to_gray:
                self.img_space = Box(low=img_min + zero_img,
                                     high=img_max + zero_img,
                                     )
            else:
                self.img_space = Box(low=img_min + zero_img,
                                     high=img_max + zero_img,
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
        if hasattr(self.env, "info_sample"):
            self.info_sample.update(self.env.info_sample)



        # Episode information
        self.episode_steps = 0
        self.total_steps = 0

    def render(self, mode='human', close=False):
        if self.render_lock is not None:
            self.render_lock.acquire()
        result = self.env.render(mode=mode, close=close)
        if self.render_lock is not None:
            self.render_lock.release()
        return result

    def preprocess_observation(self, obs):
        final = obs
        if self.new_img_size is not None:
            final = cv2.resize(obs, self.new_img_size,
                               interpolation=cv2.INTER_LINEAR)
        if self.rgb_to_gray:
            final = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
        # if len(final.shape) == 3 and final.shape[2] == 1:
        #     final = np.squeeze(final)
        return final

    def env_step(self, a):
        state_observation, reward, done, info = self.env.step(a)
        state_observation = self.s_transform(state_observation,self.total_steps)
        if self.append_image:
            if self.dummy_imagge:
                image_observation = np.random.randint(0, 255, self.img_space.shape, dtype=np.uint8)
            else:
                image_observation = self.render(mode="rgb_array")
                image_observation = self.preprocess_observation(image_observation)
            self.x_buffer = (self.x_buffer + 1) % self.num_frame
            if self.num_frame > 1:
                self.frame_buffer[:, :, self.x_buffer] = image_observation
                stacked_obs = np.take(self.frame_buffer,
                                      np.arange(self.x_buffer + 1 - self.num_frame, self.x_buffer + 1),
                                      axis=2,
                                      mode='wrap')
            else:
                stacked_obs = image_observation
            if len(stacked_obs.shape) == 2:
                stacked_obs = stacked_obs[:, :, np.newaxis]
            return [state_observation[self.vs_id], stacked_obs], reward, done, info
        else:
            return [state_observation[self.vs_id]], reward, done, info

    def env_reset(self):
        state_observation = self.env.reset()
        state_observation = self.s_transform(state_observation,self.total_steps)
        if self.append_image:
            self.frame_buffer[:] = 0
            self.x_buffer = self.num_frame - 1
            if self.dummy_imagge:
                image_observation = np.random.randint(0, 255, self.img_space.shape, dtype=np.uint8)
            else:
                image_observation = self.render(mode="rgb_array")
                image_observation = self.preprocess_observation(image_observation)
            if self.num_frame > 1:
                self.frame_buffer[:, :, self.x_buffer] = image_observation
                stacked_obs = np.take(self.frame_buffer,
                                      np.arange(self.x_buffer + 1 - self.num_frame, self.x_buffer + 1),
                                      axis=2,
                                      mode='wrap')
            else:
                stacked_obs = image_observation
            if len(stacked_obs.shape) == 2:
                stacked_obs = stacked_obs[:, :, np.newaxis]
            return [state_observation[self.vs_id], stacked_obs]
        else:
            return [state_observation[self.vs_id]]

    def step(self, a):
        final_done = False
        final_reward = 0

        final_observation, reward, done, env_info = self.env_step(a)
        full_info = {"terminated": False, **env_info}
        final_done = final_done or done
        final_reward += reward
        if done:
            full_info.update({"terminated": True})

        self.episode_steps += 1


        if self.episode_steps >= self.max_episode_length:
            full_info.update({"terminated": True})  # let the clipped episode be terminated
            final_done = True

        return final_observation, final_reward, final_done, full_info

    def _set_env_initial_state(self, s):
        assert hasattr(self.env, "initial_state")
        self.env.initial_sate = s

    def close(self):
        self.env.close()

    def sample_initial_state(self):
        pass

    def reset(self):
        observation = self.env_reset()
        self.total_steps += self.episode_steps
        self.episode_steps = 0
        return observation

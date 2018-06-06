import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py

class CustomPend(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        reward = 1.0 if not notdone else 1.0 + 1.0 * (0.2 - np.abs(ob[1])) + (3.0 - np.square(a).sum()) / 9.0
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self._get_viewer().render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().loop_once()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(init_height=64, init_width=64, visible=False)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer
# class CustomAnt(mujoco_env.MujocoEnv, utils.EzPickle):
#     FILE = 'ant.xml'
#
#     def __init__(self, file_path='ant.xml', frame_skip=5):
#         mujoco_env.MujocoEnv.__init__(self, file_path, frame_skip=frame_skip)
#         utils.EzPickle.__init__(self)
#
#     def _step(self, a):
#         xposbefore = self.get_body_com("torso")[0]
#         self.do_simulation(a, self.frame_skip)
#         xposafter = self.get_body_com("torso")[0]
#         forward_reward = (xposafter - xposbefore) / self.dt
#         ctrl_cost = .5 * np.square(a).sum()
#         contact_cost = 0.5 * 1e-3 * np.sum(
#             np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
#         survive_reward = 1.0
#         reward = forward_reward - ctrl_cost - contact_cost + survive_reward
#         state = self.state_vector()
#         notdone = np.isfinite(state).all() \
#                   and state[2] >= 0.2 and state[2] <= 1.0
#         done = not notdone
#         ob = self._get_obs()
#         return ob, reward, done, dict(
#             reward_forward=forward_reward,
#             reward_ctrl=-ctrl_cost,
#             reward_contact=-contact_cost,
#             reward_survive=survive_reward)
#
#     def _get_obs(self):
#         return np.concatenate([
#             self.model.data.qpos.flat[2:],
#             self.model.data.qvel.flat,
#             np.clip(self.model.data.cfrc_ext, -1, 1).flat,
#         ])
#
#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
#         qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
#         self.set_state(qpos, qvel)
#         return self._get_obs()
#
#     def viewer_setup(self):
#         self.viewer.cam.distance = self.model.stat.extent * 0.5
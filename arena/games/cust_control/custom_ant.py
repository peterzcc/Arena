import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class CustomAnt(mujoco_env.MujocoEnv, utils.EzPickle):
    FILE = 'ant.xml'

    def __init__(self, file_path='ant.xml', frame_skip=5):
        mujoco_env.MujocoEnv.__init__(self, file_path, frame_skip=frame_skip)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        pos_t = self.get_body_com("torso")
        xposbefore = pos_t[0]
        self.do_simulation(a, self.frame_skip)
        pos_t_prime = self.get_body_com("torso")
        xposafter = pos_t_prime[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        rot_angle = self.data.xmat[1, 8]
        notdone = np.isfinite(state).all() \
                  and rot_angle > 0 and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

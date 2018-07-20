import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjModel, MjViewer, mjcore, \
    glfw
from mujoco_py.mjlib import mjlib
from ctypes import byref
import ctypes
from threading import Lock
from arena.games.cust_control.gather_viewer import assign_curr, mjextra_append_objects, EmbeddedViewer, GatherViewer

mjCAT_ALL = 7

APPLE = 0
BOMB = 1


def x_forward_obj():
    return np.array((1, 0))


class ConstDirSingleGatherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    ORI_IND = 6

    def __init__(
            self,
            activity_range=6.,
            f_gen_obj=x_forward_obj,
            frame_skip=5,
            file_path='ant.xml',
            with_state_task=True,
            use_sparse_reward=False,
            reset_goal_prob=0,
            forward_scale=1.0,
            catch_range=0.5,
            obj_dist=1.25,
            init_noise=0.1,
            use_internal_reward=True,
            subtask_dirs=None,
            constraint_height=True,
            cam_scale=0.6,
            obj_max_dist=2.0,  # 2.5
            regenerate_goals=False,
            goal_reward=1.0,
            *args, **kwargs
    ):
        self.n_apples = 1
        self.n_bombs = 0
        self.activity_range = activity_range
        self.f_gen_obj = f_gen_obj
        self.object = np.zeros((3,), dtype=np.float32)
        self.with_state_task = with_state_task
        self.use_sparse_reward = use_sparse_reward
        self.fix_goal = use_sparse_reward
        self.reset_goal_prob = reset_goal_prob
        self.forward_scale = forward_scale
        self.catch_range = catch_range
        self.obj_dist = obj_dist
        self.subtasks_dirs = subtask_dirs
        self.use_internal_reward = use_internal_reward
        self.init_noise = init_noise
        if subtask_dirs is None:
            self.dirs = np.array([[0.0, 0.0]])
            self.info_sample = {}
        else:
            self.dirs = np.concatenate([np.array([[0.0, 0.0]]), self.subtasks_dirs], axis=0)
            self.info_sample = {"subrewards": np.zeros(self.subtasks_dirs.shape[0], np.float32)}
        self.pos_t = None
        self.pos_t_prime = None
        self.rot_angle = None
        self.constraint_height = constraint_height
        self.initial_state = None
        self.initial_pos = None
        self.cam_scale = cam_scale
        self.obs_max_dist = obj_max_dist
        self.regenerate_goals = regenerate_goals
        self.goal_reward = goal_reward
        self._reset_dir()
        mujoco_env.MujocoEnv.__init__(self, file_path, frame_skip=frame_skip)
        utils.EzPickle.__init__(self)

    def _reset(self):
        self._reset_dir()
        self.initial_pos = None
        super(ConstDirSingleGatherEnv, self)._reset()
        return self._get_obs()

    def _reset_dir(self):
        direction = self.f_gen_obj()
        self.dirs[0, :] = direction / np.sqrt(direction.dot(direction))
        self.object[0:2] = self.obj_dist * self.dirs[0]

        # assert np.any(self.dirs[0])
        # self.object = np.zeros((3,), dtype=np.float32)

    def _get_obs(self):
        # return sensor data along with data about itself
        self_obs = self._get_ant_obs()
        return np.concatenate([self_obs, ])

    def compute_cont_reward(self, a, pos_t_prime, pos_t, target_dirs, cfrc_ext):
        pos_diff = pos_t_prime[0:2] - pos_t[0:2]
        fw_dist = np.dot(target_dirs, pos_diff)
        forward_reward = fw_dist / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(cfrc_ext, -1, 1)))
        survive_reward = 1.0
        internal_reward = survive_reward - ctrl_cost - contact_cost if self.use_internal_reward else 0.0
        rewards = self.forward_scale * forward_reward + internal_reward
        return rewards[0], {"subrewards": rewards[1:]}

    def update_object_position(self, pos):
        direction = self.dirs[0]
        pos2d = pos[:2]
        if self.fix_goal:
            initial_pos_to_ant = pos2d - self.initial_pos[:2]
            # direction_norm = np.sqrt(direction.dot(direction))
            current_dist = np.dot(initial_pos_to_ant, direction)
            remaining_dist = self.obj_dist - current_dist
            if remaining_dist >= self.obs_max_dist:
                obj_pos = pos2d + self.obs_max_dist * direction
            else:
                obj_pos = pos2d + remaining_dist * direction
            self.object[0:2] = obj_pos
        else:
            obj_pos = pos2d + self.obj_dist * direction
            self.object = np.concatenate([obj_pos, (0,)])

    def _step(self, action):
        if self.initial_pos is None:
            self.set_current_as_initial_pos()
        obs, in_rw, done, info_ant = self._ant_step(action)
        if done:
            return self._get_obs(), -10, done, info_ant
        reward = in_rw

        pos = self.get_body_com("torso")
        if self.use_sparse_reward:
            if np.sum((pos[:2] - self.object[:2]) ** 2) < self.catch_range ** 2:
                reward = self.goal_reward
                if self.regenerate_goals:
                    self._reset_dir()
                    self.set_current_as_initial_pos()
                else:
                    done = True
            # elif np.max(np.abs(com[:2])) > 5:
            #     reward = -10
            #     done = True
            else:
                reward = -0.01
        if self.reset_goal_prob != 0:
            assert not self.fix_goal
            if np.random.rand() < self.reset_goal_prob:
                self._reset_dir()
                self.set_current_as_initial_pos()

        self.update_object_position(pos)
        return obs, reward, done, info_ant

    def check_if_ant_crash(self, state, rot_angle):
        is_height_legal = 0.2 <= state[2] <= 1.0 if self.constraint_height else 0.01 <= state[2] <= 5.0
        notdone = np.isfinite(state).all() \
                  and rot_angle > 0 \
                  and is_height_legal
        # TODO: verify what happens if remove the last condition
        done = not notdone
        return done

    def _ant_step(self, a):
        self.pos_t = self.get_body_com("torso")
        self.do_simulation(a, self.frame_skip)
        self.pos_t_prime = self.get_body_com("torso")
        self.state = self.state_vector()
        self.rot_angle = self.data.xmat[1, 8]
        done = self.check_if_ant_crash(self.state, self.rot_angle)
        ob = self._get_obs()
        reward, ant_info = self.compute_cont_reward(a, self.pos_t_prime, self.pos_t, self.dirs,
                                                    self.model.data.cfrc_ext)
        return ob, reward, done, ant_info

    def _get_ant_obs(self):
        if self.with_state_task:
            return np.concatenate([
                self.dirs[0],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ])
        else:
            return np.concatenate([
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ])

    def set_current_as_initial_pos(self):
        self.initial_pos = self.get_body_com("torso")[0:2].copy()

    def reset_model(self):
        if self.initial_state is None:
            qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq,
                                                           low=-0.1, high=0.1)
            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * self.init_noise
            self.set_state(qpos, qvel)
            self.initial_state = None
        else:
            qpos = self.initial_state[:self.model.nq]
            qvel = self.initial_state[self.model.nq:]
            self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_viewer(self, visible=True, init_height=500, init_width=500):
        if self.viewer is None:
            self.viewer = GatherViewer(self, init_height=init_height, init_width=init_height, visible=visible)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def viewer_setup(self):
        distance = self.model.stat.extent * self.cam_scale  # verify 0.6
        self.viewer.cam.distance = distance

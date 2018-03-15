import math
import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
from ctypes import byref
import ctypes
from ctypes import pointer, byref
import numpy as np
from custom_ant import CustomAnt
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjModel, MjViewer, mjcore, \
    mjextra, glfw
from mujoco_py.mjlib import mjlib
from ctypes import byref
import ctypes
from threading import Lock

mjCAT_ALL = 7


class EmbeddedViewer(object):
    def __init__(self):
        self.last_render_time = 0
        self.objects = mjcore.MJVOBJECTS()
        self.cam = mjcore.MJVCAMERA()
        self.vopt = mjcore.MJVOPTION()
        self.ropt = mjcore.MJROPTION()
        self.con = mjcore.MJRCONTEXT()
        self.running = False
        self.speedtype = 1
        self.window = None
        self.model = None
        self.gui_lock = Lock()

        self.last_button = 0
        self.last_click_time = 0
        self.button_left_pressed = False
        self.button_middle_pressed = False
        self.button_right_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.frames = []

    def set_model(self, model):
        self.model = model
        if model:
            self.data = model.data
        else:
            self.data = None
        if self.running:
            if model:
                mjlib.mjr_makeContext(model.ptr, byref(self.con), 150)
            else:
                mjlib.mjr_makeContext(None, byref(self.con), 150)
            self.render()
        if model:
            self.autoscale()

    def autoscale(self):
        self.cam.lookat[0] = self.model.stat.center[0]
        self.cam.lookat[1] = self.model.stat.center[1]
        self.cam.lookat[2] = self.model.stat.center[2]
        self.cam.distance = 1.0 * self.model.stat.extent
        self.cam.camid = -1
        self.cam.trackbodyid = -1
        if self.window:
            width, height = glfw.get_framebuffer_size(self.window)
            mjlib.mjv_updateCameraPose(byref(self.cam), width * 1.0 / height)

    def get_rect(self):
        rect = mjcore.MJRRECT(0, 0, 0, 0)
        rect.width, rect.height = glfw.get_framebuffer_size(self.window)
        return rect

    def record_frame(self, **kwargs):
        self.frames.append({'pos': self.model.data.qpos, 'extra': kwargs})

    def clear_frames(self):
        self.frames = []

    def render(self):
        rect = self.get_rect()
        arr = (ctypes.c_double * 3)(0, 0, 0)
        mjlib.mjv_makeGeoms(
            self.model.ptr, self.data.ptr, byref(self.objects),
            byref(self.vopt), mjCAT_ALL, 0, None, None,
            ctypes.cast(arr, ctypes.POINTER(ctypes.c_double)))
        # mjlib.mjv_setCamera(self.model.ptr, self.data.ptr, byref(self.cam))
        # mjlib.mjv_updateCameraPose(
        #     byref(self.cam), rect.width * 1.0 / rect.height)
        # mjlib.mjr_render(0, rect, byref(self.objects), byref(
        #     self.ropt), byref(self.cam.pose), byref(self.con))

    def render_internal(self):
        if not self.data:
            return
        self.gui_lock.acquire()
        self.render()

        self.gui_lock.release()

    def start(self, window):
        self.running = True

        width, height = glfw.get_framebuffer_size(window)
        width1, height = glfw.get_window_size(window)
        self.scale = width * 1.0 / width1

        self.window = window

        mjlib.mjv_makeObjects(byref(self.objects), 1000)

        mjlib.mjv_defaultCamera(byref(self.cam))
        mjlib.mjv_defaultOption(byref(self.vopt))
        mjlib.mjr_defaultOption(byref(self.ropt))

        mjlib.mjr_defaultContext(byref(self.con))

        if self.model:
            mjlib.mjr_makeContext(self.model.ptr, byref(self.con), 150)
            self.autoscale()
        else:
            mjlib.mjr_makeContext(None, byref(self.con), 150)

    def handle_mouse_move(self, window, xpos, ypos):

        # no buttons down: nothing to do
        if not self.button_left_pressed \
                and not self.button_middle_pressed \
                and not self.button_right_pressed:
            return

        # compute mouse displacement, save
        dx = int(self.scale * xpos) - self.last_mouse_x
        dy = int(self.scale * ypos) - self.last_mouse_y
        self.last_mouse_x = int(self.scale * xpos)
        self.last_mouse_y = int(self.scale * ypos)

        # require model
        if not self.model:
            return

        # get current window size
        width, height = glfw.get_framebuffer_size(self.window)

        # get shift key state
        mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS \
                    or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        # determine action based on mouse button
        action = None
        # if self.button_right_pressed:
        #     action = C.MOUSE_MOVE_H if mod_shift else C.MOUSE_MOVE_V
        # elif self.button_left_pressed:
        #     action = C.MOUSE_ROTATE_H if mod_shift else C.MOUSE_ROTATE_V
        # else:
        #     action = C.MOUSE_ZOOM

        self.gui_lock.acquire()

        mjlib.mjv_moveCamera(action, dx, dy, byref(self.cam), width, height)

        self.gui_lock.release()

    def handle_mouse_button(self, window, button, act, mods):
        # update button state
        self.button_left_pressed = \
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle_pressed = \
            glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right_pressed = \
            glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

        # update mouse position
        x, y = glfw.get_cursor_pos(window)
        self.last_mouse_x = int(self.scale * x)
        self.last_mouse_y = int(self.scale * y)

        if not self.model:
            return

        self.gui_lock.acquire()

        # save info
        if act == glfw.PRESS:
            self.last_button = button
            self.last_click_time = glfw.get_time()

        self.gui_lock.release()

    def handle_scroll(self, window, x_offset, y_offset):
        # require model
        if not self.model:
            return

        # get current window size
        width, height = glfw.get_framebuffer_size(window)

        # scroll
        self.gui_lock.acquire()
        # mjlib.mjv_moveCamera(C.MOUSE_ZOOM, 0, (-20 * y_offset),
        #                      byref(self.cam), width, height)
        self.gui_lock.release()

    def should_stop(self):
        return glfw.window_should_close(self.window)

    def loop_once(self):
        self.render()
        # Swap front and back buffers
        glfw.swap_buffers(self.window)
        # Poll for and process events
        glfw.poll_events()

    def finish(self):
        glfw.terminate()
        mjlib.mjr_freeContext(byref(self.con))
        mjlib.mjv_freeObjects(byref(self.objects))
        self.running = False


APPLE = 0
BOMB = 1


class GatherViewer(MjViewer):
    def __init__(self, env, visible=True, init_width=500, init_height=500, go_fast=False):
        self.env = env
        super(GatherViewer, self).__init__(visible=visible, init_width=init_width, init_height=init_height,
                                           go_fast=go_fast)
        green_ball_model = MjModel('green_ball.xml')
        self.green_ball_renderer = EmbeddedViewer()
        self.green_ball_model = green_ball_model
        self.green_ball_renderer.set_model(green_ball_model)
        mjextra.append_objects(
            self.objects, self.green_ball_renderer.objects)

    def start(self):
        super(GatherViewer, self).start()
        self.green_ball_renderer.start(self.window)

    def handle_mouse_move(self, window, xpos, ypos):
        super(GatherViewer, self).handle_mouse_move(window, xpos, ypos)
        self.green_ball_renderer.handle_mouse_move(window, xpos, ypos)
    def handle_scroll(self, window, x_offset, y_offset):
        super(GatherViewer, self).handle_scroll(window, x_offset, y_offset)
        self.green_ball_renderer.handle_scroll(window, x_offset, y_offset)

    def render(self):
        if not self.data:
            return

        self.gui_lock.acquire()
        glfw.make_context_current(self.window)

        obj = self.env.objects[0]
        x, y, typ = obj
        # print x, y
        qpos = np.zeros_like(self.green_ball_model.data.qpos)
        qpos[0, 0] = x
        qpos[1, 0] = y
        if typ == APPLE:
            self.green_ball_model.data.qpos = qpos
            self.green_ball_model.forward()
            self.green_ball_renderer.render()

        rect = self.get_rect()
        arr = (ctypes.c_double * 3)(0, 0, 0)

        mjlib.mjv_makeGeoms(self.model.ptr, self.data.ptr, byref(self.objects), byref(self.vopt), mjCAT_ALL, 0, None,
                            None, ctypes.cast(arr, ctypes.POINTER(ctypes.c_double)))
        # mjlib.mjv_makeLights(self.model.ptr, self.data.ptr, byref(self.objects))
        #
        # mjlib.mjv_setCamera(self.model.ptr, self.data.ptr, byref(self.cam))
        #
        # mjlib.mjv_updateCameraPose(byref(self.cam), rect.width*1.0/rect.height)
        #
        # mjlib.mjr_render(0, rect, byref(self.objects), byref(self.ropt), byref(self.cam.pose), byref(self.con))



        tmpobjects = mjcore.MJVOBJECTS()
        mjlib.mjv_makeObjects(byref(tmpobjects), 1000)
        mjextra.append_objects(
            tmpobjects, self.green_ball_renderer.objects)
        mjextra.append_objects(tmpobjects, self.objects)
        mjlib.mjv_makeLights(
            self.model.ptr, self.data.ptr, byref(tmpobjects))
        mjlib.mjv_setCamera(self.model.ptr, self.data.ptr, byref(self.cam))

        mjlib.mjv_updateCameraPose(byref(self.cam), rect.width * 1.0 / rect.height)
        mjlib.mjr_render(0, self.get_rect(), byref(tmpobjects), byref(
            self.ropt), byref(self.cam.pose), byref(self.con))

        self.gui_lock.release()
        mjlib.mjv_freeObjects(byref(tmpobjects))


def x_forward_obj():
    return np.array((1, 0))


class SingleGatherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    ORI_IND = 6
    OBJ_DIST = 1.25

    def __init__(
            self,
            activity_range=6.,
            f_gen_obj=x_forward_obj,
            frame_skip=5,
            file_path='ant.xml',
            with_state_task=True,
            use_internal_reward=True,
            reset_goal_prob=0,
            *args, **kwargs
    ):
        self.n_apples = 1
        self.n_bombs = 0
        self.activity_range = activity_range
        self.f_gen_obj = f_gen_obj
        self.dir = None
        self.objects = []
        self.with_state_task = with_state_task
        self.use_internal_reward = use_internal_reward
        self.reset_goal_prob = reset_goal_prob

        self._reset_objects()
        mujoco_env.MujocoEnv.__init__(self, file_path, frame_skip=frame_skip)
        utils.EzPickle.__init__(self)

    def _reset(self):
        self._reset_objects()
        super(SingleGatherEnv, self)._reset()
        return self._get_obs()

    def _reset_objects(self):
        self.dir = self.f_gen_obj()
        self.objects = [np.concatenate([self.OBJ_DIST * self.dir, (0,)])]

    def _get_obs(self):
        # return sensor data along with data about itself
        self_obs = self._get_ant_obs()
        return np.concatenate([self_obs, ])

    def _step(self, action):
        obs, in_rw, done, info = self._ant_step(action)
        if done:
            return self._get_obs(), -10, done, info
        com = self.get_body_com("torso")
        x, y = com[:2]
        obj_pos = com[:2] + self.OBJ_DIST * self.dir
        self.objects[0] = np.concatenate([obj_pos, (0,)])
        if self.reset_goal_prob != 0:
            if np.random.rand() < self.reset_goal_prob:
                self._reset_objects()
        return obs, in_rw, done, info

    def _ant_step(self, a):
        pos_t = self.get_body_com("torso")
        self.do_simulation(a, self.frame_skip)
        pos_t_prime = self.get_body_com("torso")
        fw_dist = np.dot(pos_t_prime[0:2] - pos_t[0:2], self.dir)
        forward_reward = fw_dist / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        internal_reward = - ctrl_cost - contact_cost if self.use_internal_reward else 0
        reward = 10.0 * forward_reward + survive_reward + internal_reward
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

    def _get_ant_obs(self):
        if self.with_state_task:
            return np.concatenate([
                self.dir,
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

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
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
        self.viewer.cam.distance = self.model.stat.extent * 0.6


class SimpleSingleGatherEnv(SingleGatherEnv):
    slc_dict = [((0, 32), (0, 32)),
                ((32, 64), (0, 32)),
                ((0, 32), (32, 64)),
                ((32, 64), (32, 64)), ]
    cord_to_id = {1: {0: 0}, 0: {1: 1, -1: 3}, -1: {0: 2}}

    def render(self, mode='human', close=False):
        assert mode == "rgb_array" or close == True
        blank = np.zeros((64, 64, 1), dtype='uint8')
        task_id = self.cord_to_id[self.dir[0]][self.dir[1]]
        slc = self.slc_dict[task_id]
        blank[slc[0][0]:slc[0][1], slc[1][0]:slc[1][1], :] = 255
        return blank

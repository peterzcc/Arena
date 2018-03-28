import math
import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
from ctypes import byref

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
        mjlib.mjv_setCamera(self.model.ptr, self.data.ptr, byref(self.cam))
        mjlib.mjv_updateCameraPose(
            byref(self.cam), rect.width * 1.0 / rect.height)
        mjlib.mjr_render(0, rect, byref(self.objects), byref(
            self.ropt), byref(self.cam.pose), byref(self.con))

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
        red_ball_model = MjModel('red_ball.xml')
        self.red_ball_renderer = EmbeddedViewer()
        self.red_ball_model = red_ball_model
        self.red_ball_renderer.set_model(red_ball_model)

    def start(self):
        super(GatherViewer, self).start()
        self.green_ball_renderer.start(self.window)
        self.red_ball_renderer.start(self.window)

    def handle_mouse_move(self, window, xpos, ypos):
        super(GatherViewer, self).handle_mouse_move(window, xpos, ypos)
        self.green_ball_renderer.handle_mouse_move(window, xpos, ypos)
        self.red_ball_renderer.handle_mouse_move(window, xpos, ypos)

    def handle_scroll(self, window, x_offset, y_offset):
        super(GatherViewer, self).handle_scroll(window, x_offset, y_offset)
        self.green_ball_renderer.handle_scroll(window, x_offset, y_offset)
        self.red_ball_renderer.handle_scroll(window, x_offset, y_offset)

    def render(self):
        super(GatherViewer, self).render()
        tmpobjects = mjcore.MJVOBJECTS()
        mjlib.mjv_makeObjects(byref(tmpobjects), 1000)
        for obj in self.env.objects:
            x, y, typ = obj
            # print x, y
            qpos = np.zeros_like(self.green_ball_model.data.qpos)
            qpos[0, 0] = x
            qpos[1, 0] = y
            if typ == APPLE:
                self.green_ball_model.data.qpos = qpos
                self.green_ball_model.forward()
                self.green_ball_renderer.render()
                mjextra.append_objects(
                    tmpobjects, self.green_ball_renderer.objects)
            else:
                self.red_ball_model.data.qpos = qpos
                self.red_ball_model.forward()
                self.red_ball_renderer.render()
                mjextra.append_objects(
                    tmpobjects, self.red_ball_renderer.objects)
        mjextra.append_objects(tmpobjects, self.objects)
        mjlib.mjv_makeLights(
            self.model.ptr, self.data.ptr, byref(tmpobjects))
        mjlib.mjr_render(0, self.get_rect(), byref(tmpobjects), byref(
            self.ropt), byref(self.cam.pose), byref(self.con))

        try:
            import OpenGL.GL as GL
        except:
            return

        def draw_rect(x, y, width, height):
            # start drawing a rectangle
            GL.glBegin(GL.GL_QUADS)
            # bottom left point
            GL.glVertex2f(x, y)
            # bottom right point
            GL.glVertex2f(x + width, y)
            # top right point
            GL.glVertex2f(x + width, y + height)
            # top left point
            GL.glVertex2f(x, y + height)
            GL.glEnd()

        def refresh2d(width, height):
            GL.glViewport(0, 0, width, height)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GL.glOrtho(0.0, width, 0.0, height, 0.0, 1.0)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

        GL.glLoadIdentity()
        width, height = glfw.get_framebuffer_size(self.window)
        refresh2d(width, height)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_BLEND)

        GL.glColor4f(0.0, 0.0, 0.0, 0.8)
        draw_rect(10, 10, 300, 100)

        apple_readings, bomb_readings = self.env._get_readings()
        for idx, reading in enumerate(apple_readings):
            if reading > 0:
                GL.glColor4f(0.0, 1.0, 0.0, reading)
                draw_rect(20 * (idx + 1), 10, 5, 50)
        for idx, reading in enumerate(bomb_readings):
            if reading > 0:
                GL.glColor4f(1.0, 0.0, 0.0, reading)
                draw_rect(20 * (idx + 1), 60, 5, 50)


class GatherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    MODEL_CLASS = CustomAnt
    ORI_IND = 6

    def __init__(
            self,
            n_apples=8,
            n_bombs=0,
            activity_range=6.,
            robot_object_spacing=2.,
            catch_range=1.,
            n_bins=10,
            sensor_range=6.,
            sensor_span=math.pi,
            frame_skip=5,
            *args, **kwargs
    ):
        self.n_apples = n_apples
        self.n_bombs = n_bombs
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.objects = []
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise ValueError("MODEL_CLASS unspecified!")

        xml_path = model_cls.FILE
        file_path = self.generate_full_xml(xml_path=xml_path)
        self._reset_objects()
        mujoco_env.MujocoEnv.__init__(self, file_path, frame_skip=frame_skip)
        utils.EzPickle.__init__(self)

    def generate_full_xml(self, xml_path):
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")
        attrs = dict(
            type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3"
        )
        walldist = self.activity_range + 1
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall1",
                pos="0 -%d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall2",
                pos="0 %d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall3",
                pos="-%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall4",
                pos="%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path)
        return file_path

    def _reset(self):
        self._reset_objects()
        super(GatherEnv, self)._reset()
        return self._get_obs()

    def _reset_objects(self):
        self.objects = []
        existing = set()
        while len(self.objects) < self.n_apples:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = APPLE
            self.objects.append((x, y, typ))
            existing.add((x, y))
        while len(self.objects) < self.n_apples + self.n_bombs:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = BOMB
            self.objects.append((x, y, typ))
            existing.add((x, y))

    def _get_obs(self):
        # return sensor data along with data about itself
        self_obs = self._get_ant_obs()
        apple_readings, bomb_readings = self._get_readings()
        return np.concatenate([self_obs, apple_readings, bomb_readings])

    def _get_readings(self):
        # compute sensor readings
        # first, obtain current orientation
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins
        ori = self.model.data.qpos[self.__class__.ORI_IND]
        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb
                ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res) - 1
            #    ((angle + half_span) +
            #     ori) % (2 * math.pi) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity
        return apple_readings, bomb_readings

    def _step(self, action):
        _, _, done, info = self._ant_step(action)
        if done:
            return self._get_obs(), -10, done, info
        com = self.get_body_com("torso")
        x, y = com[:2]
        reward = 0
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward = reward + 1
                else:
                    reward = reward - 1
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0
        return self._get_obs(), reward, done, info

    def _ant_step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_ant_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_ant_obs(self):
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
        return self.viewer

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

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


def line_intersect(pt1, pt2, ptA, ptB):
    """
    Taken from https://www.cs.hmc.edu/ACM/lectures/intersections.html

    this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

    returns a tuple: (xi, yi, valid, r, s), where
    (xi, yi) is the intersection
    r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
    s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
    valid == 0 if there are 0 or inf. intersections (invalid)
    valid == 1 if it has a unique intersection ON the segment
    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0, 0, 0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (xi, yi, 1, r, s)


def ray_segment_intersect(ray, segment):
    """
    Check if the ray originated from (x, y) with direction theta intersects the line segment (x1, y1) -- (x2, y2),
    and return the intersection point if there is one
    """
    (x, y), theta = ray
    # (x1, y1), (x2, y2) = segment
    pt1 = (x, y)
    len = 1
    pt2 = (x + len * math.cos(theta), y + len * math.sin(theta))
    xo, yo, valid, r, s = line_intersect(pt1, pt2, *segment)
    if valid and r >= 0 and 0 <= s <= 1:
        return (xo, yo)
    return None


def point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    MODEL_CLASS = CustomAnt
    ORI_IND = 6
    MAZE_HEIGHT = 0.02
    MAZE_SIZE_SCALING = 3.0

    MAZE_MAKE_CONTACTS = False
    MAZE_STRUCTURE = [
        [1, 1, 1, 1, 1],
        [1, 'r', 0, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 'g', 0, 1, 1],
        [1, 1, 1, 1, 1],
    ]

    MANUAL_COLLISION = False

    def __init__(
            self,
            n_bins=20,
            sensor_range=10.,
            sensor_span=math.pi,
            frame_skip=5,
            *args,
            **kwargs):

        self._n_bins = n_bins
        self._sensor_range = sensor_range
        self._sensor_span = sensor_span

        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise ValueError("MODEL_CLASS unspecified!")
        xml_path = model_cls.FILE
        file_path = self.generate_full_xml(xml_path)

        self._goal_range = self._find_goal_range()
        self._cached_segments = None

        mujoco_env.MujocoEnv.__init__(self, file_path, frame_skip=frame_skip)
        utils.EzPickle.__init__(self)

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
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
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

    def generate_full_xml(self, xml_path):
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        size_scaling = self.__class__.MAZE_SIZE_SCALING
        height = self.__class__.MAZE_HEIGHT
        structure = self.__class__.MAZE_STRUCTURE

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) == '1':
                    # offset all coordinates so that robot starts at the origin
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1"
                    )
        ball_body = ET.SubElement(
            worldbody, "body",
            pos="%f %f %f" % (1 * size_scaling - torso_x,
                              1.25 * size_scaling - torso_y,
                              0.125)
        )

        ball_geom = ET.SubElement(
            ball_body, "geom",
            name="ball",
            size="0.25",
            type="sphere",
            material="",
            contype="1",
            conaffinity="1",
            rgba="1.0 0.0 0.0 1"
        )
        ET.SubElement(
            ball_body, "joint",
            armature="0",
            damping="0",
            limited="false",
            margin="0.01",
            name="rootball",
            type="free"
        )
        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        if self.__class__.MAZE_MAKE_CONTACTS:
            contact = ET.SubElement(
                tree.find("."), "contact"
            )
            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if str(structure[i][j]) == '1':
                        for geom in geoms:
                            ET.SubElement(
                                contact, "pair",
                                geom1=geom.attrib["name"],
                                geom2="block_%d_%d" % (i, j)
                            )

        _, file_path = tempfile.mkstemp(text=True)
        # file_path = "ant_maze.xml"
        tree.write(file_path)
        return file_path

    def _get_readings(self):
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment
        robot_x, robot_y = self.get_body_com("torso")[:2]
        ori = self.model.data.qpos[self.__class__.ORI_IND]

        # print ori

        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING

        segments = []
        # compute the distance of all segments

        # Get all line segments of the goal and the obstacles
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1 or structure[i][j] == 'g':
                    cx = j * size_scaling - self._init_torso_x
                    cy = i * size_scaling - self._init_torso_y
                    x1 = cx - 0.5 * size_scaling
                    x2 = cx + 0.5 * size_scaling
                    y1 = cy - 0.5 * size_scaling
                    y2 = cy + 0.5 * size_scaling
                    struct_segments = [
                        ((x1, y1), (x2, y1)),
                        ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)),
                        ((x1, y2), (x1, y1)),
                    ]
                    for seg in struct_segments:
                        segments.append(dict(
                            segment=seg,
                            type=structure[i][j],
                        ))

        wall_readings = np.zeros(self._n_bins)
        goal_readings = np.zeros(self._n_bins)

        for ray_idx in range(self._n_bins):
            ray_ori = ori - self._sensor_span * 0.5 + 1.0 * (2 * ray_idx + 1) / (2 * self._n_bins) * self._sensor_span
            ray_segments = []
            for seg in segments:
                p = ray_segment_intersect(ray=((robot_x, robot_y), ray_ori), segment=seg["segment"])
                if p is not None:
                    ray_segments.append(dict(
                        segment=seg["segment"],
                        type=seg["type"],
                        ray_ori=ray_ori,
                        distance=point_distance(p, (robot_x, robot_y)),
                    ))
            if len(ray_segments) > 0:
                first_seg = sorted(ray_segments, key=lambda x: x["distance"])[0]
                # print first_seg
                if first_seg["type"] == 1:
                    # Wall -> add to wall readings
                    if first_seg["distance"] <= self._sensor_range:
                        wall_readings[ray_idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range
                elif first_seg["type"] == 'g':
                    # Goal -> add to goal readings
                    if first_seg["distance"] <= self._sensor_range:
                        goal_readings[ray_idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range
                else:
                    assert False

        obs = np.concatenate([
            wall_readings,
            goal_readings
        ])
        return obs

    def _get_obs(self):

        # print "wall readings:", wall_readings
        # print "goal readings:", goal_readings
        obs = self._get_ant_obs()
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _find_robot(self):
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    return j * size_scaling, i * size_scaling
        assert False

    def _find_goal_range(self):
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'g':
                    minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
                    maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
                    miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
                    maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
                    return minx, maxx, miny, maxy

    # def _is_in_collision(self, pos):
    #     x, y = pos
    #     structure = self.__class__.MAZE_STRUCTURE
    #     size_scaling = self.__class__.MAZE_SIZE_SCALING
    #     for i in range(len(structure)):
    #         for j in range(len(structure[0])):
    #             if structure[i][j] == 1:
    #                 minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
    #                 maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
    #                 miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
    #                 maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
    #                 if minx <= x <= maxx and miny <= y <= maxy:
    #                     return True
    #     return False

    def _step(self, action):
        if not self.MANUAL_COLLISION:
            _, _, done, info = self._ant_step(action)
        else:
            done, info = (False, {})
            raise ValueError("Not implemented")
        next_obs = self._get_obs()
        x, y = self.get_body_com("torso")[:2]
        # ref_x = x + self._init_torso_x
        # ref_y = y + self._init_torso_y
        reward = 0
        minx, maxx, miny, maxy = self._goal_range
        # print "goal range: x [%s,%s], y [%s,%s], now [%s,%s]" % (str(minx), str(maxx), str(miny), str(maxy),
        #                                                          str(x), str(y))
        if minx <= x <= maxx and miny <= y <= maxy:
            done = True
            reward = 1
        return (next_obs, reward, done, info)


        # def action_from_key(self, key):
        #     return self.wrapped_env.action_from_key(key)

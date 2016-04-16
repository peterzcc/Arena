__author__ = 'flyers'

import logging
import numpy
from .game import Game
from arena.utils import *
from arena import ReplayMemory
from .game import DEFAULT_MAX_EPISODE_STEP

try:
    from arena.games.vrep import vrep
except ImportError as e:
    print "Failed to import vrep, please refer to " \
          "http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm", e.message

logger = logging.getLogger(__name__)

'''
this class uses the remote api of V-REP to wrap a quadcopter controlling game, where the
goal is to autonomously control the quadcopter to follow the given target.
action: continuous float values on four motors
state: quadcopter data (position, orientation, their derivatives), target data (position, assumed unknown)
observation: raw image from the quadcopter sensor, RGB matrix
reward: calculated based on the quadcopter orientation and target location in the camera view
'''


class VREPGame(Game):
    def __init__(self, frame_skip=1, history_length=3,
                 replay_memory_size=1000000,
                 replay_start_size=100):
        super(VREPGame, self).__init__()
        self.rng = get_numpy_rng()
        self.frame_skip = frame_skip
        self.history_length = history_length

        # members about the vrep environment
        self.client_id = -1
        self.quadcopter_handle = None
        self.target_handle = None
        self.camera_handle = None
        # width by height ie column by row
        self.resolution = None
        # image observation
        # self.image = None
        # linear and angular velocity of quadcopter in the absolute frame
        self.linear_velocity_g = None
        self.angular_velocity_g = None
        # linear and angular velocity of quadcopter in the body frame
        self.linear_velocity_b = None
        self.angular_velocity_b = None
        # position and orientation of quadcopter in the absolute frame
        self.quadcopter_pos = None
        self.quadcopter_orientation = None
        self.quadcopter_quaternion = None
        # orientation velocity of quadcopter in the body frame
        self.quadcopter_angular_variation = None
        # target object absolute position
        self.target_pos = None
        # target coordinates position from the view of quadcopter camera
        self.target_coordinates = None

        self.start()

        '''
        states configuration: [linear_velocity_g, angular_velocity_g, target_coordinates]
        action: low-level motor command outputs, 4-dimensional motor velocities
        Note: in the future, we may want to directly use the raw camera image content as the state representation
        1. test stability
        2. test target following
        for now, focus on 1.
        '''
        self.replay_memory = ReplayMemory(state_dim=(6,),
                                          action_dim=(4,),
                                          state_dtype='float32', action_dtype='float32',
                                          history_length=history_length,
                                          memory_size=replay_memory_size,
                                          replay_start_size=replay_start_size)



    def _read_camera_image(self):
        _, self.resolution, self.image = vrep.simxGetVisionSensorImage(
            self.client_id, self.camera_handle, 0, vrep.simx_opmode_buffer)
        self.image = numpy.array(self.image).reshape((self.resolution[1], self.resolution[0], 3))
        self.image = numpy.flipud(self.image)
        index = numpy.zeros(self.image.shape)
        index[self.image < 0] = 1
        self.image += 256 * index


    def _read_vrep_data(self):
        # self._read_camera_image()
        _, self.linear_velocity_g, self.angular_velocity_g = vrep.simxGetObjectVelocity(
            self.client_id, self.quadcopter_handle, vrep.simx_opmode_buffer)
        _, self.quadcopter_pos = vrep.simxGetObjectPosition(
            self.client_id, self.quadcopter_handle, -1, vrep.simx_opmode_buffer)
        _, self.quadcopter_orientation = vrep.simxGetObjectOrientation(
            self.client_id, self.quadcopter_handle, -1, vrep.simx_opmode_buffer)
        _, self.target_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_handle, -1, vrep.simx_opmode_buffer)

        _, self.quadcopter_angular_variation = vrep.simxGetStringSignal(
            self.client_id, 'angular_variations', vrep.simx_opmode_buffer)
        self.quadcopter_angular_variation = vrep.simxUnpackFloats(
            self.quadcopter_angular_variation)
        _, self.quadcopter_quaternion = vrep.simxGetStringSignal(self.client_id, 'quaternion', vrep.simx_opmode_buffer)
        self.quadcopter_quaternion = vrep.simxUnpackFloats(
            self.quadcopter_quaternion)

        self.angular_velocity_b = self.quadcopter_angular_variation
        mat = self._quad2mat(self.quadcopter_quaternion)
        self.linear_velocity_b = mat.transpose().dot(self.linear_velocity_g)

    def print_vrep_data(self):
        print 'absolute linear velocity: %.4f\t%.4f\t%.4f' % (self.linear_velocity_g[0], self.linear_velocity_g[1], self.linear_velocity_g[2])
        print 'body linear velocity: %.4f\t%.4f\t%.4f' % (self.linear_velocity_b[0], self.linear_velocity_b[1], self.linear_velocity_b[2])
        print 'absolute angular velocity: %.4f\t%.4f\t%.4f' % (self.angular_velocity_g[0], self.angular_velocity_g[1], self.angular_velocity_g[2])
        print 'body angular velocity: %.4f\t%.4f\t%.4f' % (self.angular_velocity_b[0], self.angular_velocity_b[1], self.angular_velocity_b[2])
        print 'quadcopter position: %.4f\t%.4f\t%.4f' % (self.quadcopter_pos[0], self.quadcopter_pos[1], self.quadcopter_pos[2])
        print 'target position: %.4f\t%.4f\t%.4f' % (self.target_pos[0], self.target_pos[1], self.target_pos[2])



    def _quad2mat(self, q):
        mat = numpy.zeros((3, 3), dtype='float32')
        q = numpy.array(q)
        sq = q * q
        mat[0, 0] = numpy.array([1, -1, -1, 1]).dot(sq)
        mat[1, 1] = numpy.array([-1, 1, -1, 1]).dot(sq)
        mat[2, 2] = numpy.array([-1, -1, 1, 1]).dot(sq)

        xy = q[0] * q[1]
        zw = q[2] * q[3]
        mat[1, 0] = 2 * (xy + zw)
        mat[0, 1] = 2 * (xy - zw)

        xz = q[0] * q[2]
        yw = q[1] * q[3]
        mat[2, 0] = 2 * (xz - yw)
        mat[0, 2] = 2 * (xz + yw)

        yz = q[1] * q[2]
        xw = q[0] * q[3]
        mat[2, 1] = 2 * (yz + xw)
        mat[1, 2] = 2 * (yz - xw)

        return mat

    def _get_track_coordinates(self):
        # TODO: use tracking or camera transformation to get the target position on the camera image
        self.target_coordinates = numpy.array([1, 1, 1, 1])

    def _get_reward(self):
        # compute an instant reward based on quadcopter orientation velocities, target tracking position
        shaking_penalty = -0.1
        penalty_boundary = 0.6
        reward = 0.0
        reward += max(abs(self.quadcopter_angular_variation[0]) - penalty_boundary, 0) * shaking_penalty + \
                  max(abs(self.quadcopter_angular_variation[1]) - penalty_boundary, 0) * shaking_penalty

        # encourage the quadcopter keep moving
        reward += 0.1 * (abs(self.linear_velocity_b[0]) + abs(self.linear_velocity_b[1]))

        # width = self.resolution[0]
        # height = self.resolution[1]
        # # define a center area where we should keep the target in
        # center_area = numpy.round([width*0.5, height*0.7, width*0.35, height*0.35])
        # TODO tracking position part
        reward += 0
        return reward

    def start(self):
        vrep.simxFinish(self.client_id)  # just in case, close all opened connections
        self.client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        if self.client_id == -1:
            print "Failed connecting to remote API server"
            exit(0)
        # enable the synchronous mode on the client
        vrep.simxSynchronous(self.client_id, True)

        # get object handles
        _, self.quadcopter_handle = vrep.simxGetObjectHandle(
            self.client_id, 'Quadricopter_base',vrep.simx_opmode_oneshot_wait)
        _, self.target_handle = vrep.simxGetObjectHandle(
            self.client_id, 'Bill', vrep.simx_opmode_oneshot_wait)
        _, self.camera_handle = vrep.simxGetObjectHandle(
            self.client_id, 'Quadricopter_frontSensor', vrep.simx_opmode_oneshot_wait)

        # start the simulation, in blocking mode
        vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)

        # enable streaming of state values and the observation image
        # _, _, self.image = vrep.simxGetVisionSensorImage(
        #     self.client_id, self.camera_handle, 0, vrep.simx_opmode_streaming)
        _, self.linear_velocity_g, self.angular_velocity_g = vrep.simxGetObjectVelocity(
            self.client_id, self.quadcopter_handle, vrep.simx_opmode_streaming)
        _, self.quadcopter_pos = vrep.simxGetObjectPosition(
            self.client_id, self.quadcopter_handle, -1, vrep.simx_opmode_streaming)
        _, self.quadcopter_orientation = vrep.simxGetObjectOrientation(
            self.client_id, self.quadcopter_handle, -1, vrep.simx_opmode_streaming)
        _, self.target_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_handle, -1, vrep.simx_opmode_streaming)

        _, self.quadcopter_angular_variation = vrep.simxGetStringSignal(
            self.client_id, 'angular_variations', vrep.simx_opmode_streaming)
        self.quadcopter_angular_variation = vrep.simxUnpackFloats(
            self.quadcopter_angular_variation)
        _, self.quadcopter_quaternion = vrep.simxGetStringSignal(self.client_id, 'quaternion', vrep.simx_opmode_streaming)
        self.quadcopter_quaternion = vrep.simxUnpackFloats(
            self.quadcopter_quaternion)

        self.total_reward = 0
        self.episode_reward = 0
        self.episode_step = 0
        self.max_episode_step = DEFAULT_MAX_EPISODE_STEP

    def begin_episode(self, max_episode_step=DEFAULT_MAX_EPISODE_STEP):
        if self.episode_step > self.max_episode_step or self.episode_terminate():
            vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)
            self.start()
        self.max_episode_step = max_episode_step
        self.episode_reward = 0
        self.episode_step = 0

        # trigger several simulation step for initialization
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxSynchronousTrigger(self.client_id)

    def play(self, a):
        self.episode_step += 1
        reward = 0.0

        for i in xrange(self.frame_skip):
            # send control signals
            vrep.simxSetStringSignal(self.client_id, 'thrust',
                                     vrep.simxPackFloats(a), vrep.simx_opmode_oneshot)
            # trigger next simulation step
            vrep.simxSynchronousTrigger(self.client_id)
            self._read_vrep_data()
            self._get_track_coordinates()
            # get reward
            reward += self._get_reward()

        self.total_reward += reward
        self.episode_reward += reward
        # save current observation or next step observation
        ob = self.get_observation()
        terminate_flag = self.episode_terminate()
        # whether clip the reward or not
        self.replay_memory.append(ob, a, reward, terminate_flag)
        return reward, terminate_flag

    def episode_terminate(self):
        return self.episode_step > 0 and (self.quadcopter_pos[2] <= 0
               or abs(self.quadcopter_angular_variation[0]) >= 1
               or abs(self.quadcopter_angular_variation[1]) >= 1)

    def get_observation(self):
        return numpy.array(self.linear_velocity_b.tolist() + self.angular_velocity_b, dtype='float32')

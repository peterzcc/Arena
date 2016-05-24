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
    def __init__(self, remote_port=19997, frame_skip=1, history_length=1,
                 replay_memory_size=1000000,
                 replay_start_size=100):
        super(VREPGame, self).__init__()
        self.rng = get_numpy_rng()
        self.frame_skip = frame_skip
        self.history_length = history_length
        self.remote_port = remote_port

        # members about the vrep environment
        self.client_id = -1
        self.quadcopter_handle = None
        self.target_handle = None
        self.camera_handle = None
        self.target_neck = None
        self.target_back = None
        self.target_leftfoot = None
        self.target_rightfoot = None
        # width by height ie column by row
        self.resolution = [128, 128]
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
        # auxiliary variables used to compute self.target_coordinates
        self.target_neck_pos = None
        self.target_back_pos = None
        self.target_leftfoot_pos = None
        self.target_rightfoot_pos = None
        # self.action_bounds = [4.8, 6.0]
        self.action_bounds = [4.583759, 6.501942]
        # bounding box coordinates and scale are normalized to range [0,1]
        # desired [cx, cy, height]
        self.desire_goal = numpy.array([0., 0., 0.4])
        self.desire_velocity = numpy.array([2, 2, 0])

        self.start()

        '''
        states configuration: [linear_velocity_g, angular_velocity_g, target_coordinates]
        action: low-level motor command outputs, 4-dimensional motor velocities
        Note: in the future, we may want to directly use the raw camera image content as the state representation
        '''
        self.replay_memory = ReplayMemory(state_dim=(6+3,),
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
        index = numpy.zeros(self.image.shape, dtype=self.image.dtype)
        index[self.image < 0] = 1
        self.image += 256 * index
        self.image = self.image.transpose(2, 0, 1)


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

        _, self.target_neck_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_neck, self.camera_handle, vrep.simx_opmode_buffer)
        _, self.target_back_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_back, self.camera_handle, vrep.simx_opmode_buffer)
        _, self.target_leftfoot_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_leftfoot, self.camera_handle, vrep.simx_opmode_buffer)
        _, self.target_rightfoot_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_rightfoot, self.camera_handle, vrep.simx_opmode_buffer)


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

    def _transform_coordinates(self, vrep_pos):
        pos = numpy.array(
                [vrep_pos[0]/vrep_pos[2], vrep_pos[1]/vrep_pos[2]])
        pos = numpy.array(self.resolution)/2 - pos * self.resolution
        return pos

    def _get_track_coordinates(self):
        # use built in APIs in V-REP to get the target position on the camera image
        # for scale, we only consider the height information and ignore the width
        cx = self.target_back_pos[0] / self.target_back_pos[2]
        y_top = self.target_neck_pos[1] / self.target_neck_pos[2]
        y_bottom = (self.target_leftfoot_pos[1]/self.target_leftfoot_pos[2] +
                    self.target_rightfoot_pos[1]/self.target_rightfoot_pos[2]) / 2.0
        h = abs(y_bottom - y_top)
        cy = (y_bottom + y_top) / 2.0
        self.target_coordinates = numpy.array([cx, cy, h])
        # print 'target:', self.target_coordinates

    def _get_reward(self):
        # compute an instant reward based on quadcopter orientation velocities, target tracking position
        # reward = 0.0
        # velocity = abs(numpy.array(self.linear_velocity_b))
        # velocity_reward = numpy.exp(-numpy.sum((velocity - self.desire_velocity)**2))
        # reward += velocity_reward
        reward = numpy.exp(-numpy.sum((self.target_coordinates - self.desire_goal)**2))
        # maintain velocity, x,y [0, 1], z [0, 0.25]
        # velocity = abs(numpy.array(self.linear_velocity_b))
        # velocity_clip = numpy.clip(velocity, 0, [1, 1, 0.25])
        # velocity_cross = numpy.int8(velocity - velocity_clip > 0)
        # velocity_reward = numpy.sum((1 - velocity_cross) * (velocity_clip+1) * [1, 1, 0]) -\
        #                   numpy.sum(velocity_cross)
        #                   # numpy.sum(velocity_cross * (velocity - [1, 1, 0.25]))
        #
        # # maintain altitude at [1.5, 3]
        # height = self.quadcopter_pos[2]
        # height_reward = -((height < 1.5) + (height > 3))
        #
        # # maintain angular velocity, x,y [0, 0.5], do not care about yaw
        # angular = abs(numpy.array(self.angular_velocity_b))
        # angular_clip = numpy.clip(angular, 0, 0.5)
        # angular_cross = numpy.int8(angular - angular_clip > 0)
        # angular_reward = numpy.sum((1 - angular_cross) * [1, 1, 0]) - \
        #                  numpy.sum(angular_cross * [2, 2, 0])
        #                  # numpy.sum(angular_cross * (angular - 0.5) * [2, 2, 0])
        #
        # # maintain angles, x,y [0, 0.2]
        # angle = abs(numpy.array(self.quadcopter_orientation))
        # angle_clip = numpy.clip(angle, 0, 0.2)
        # # angle_clip = numpy.clip(angle, 0, 0.1)
        # angle_cross = numpy.int8(angle > angle_clip)
        # angle_reward = numpy.sum((1 - angle_cross) * [1, 1, 0]) - numpy.sum(numpy.array([1, 1, 0]) * angle_cross)
        # reward = velocity_reward + height_reward + angular_reward + angle_reward

        return reward

    def start(self):
        if self.client_id != -1:
            vrep.simxFinish(self.client_id)  # just in case, close all opened connections
        self.client_id = vrep.simxStart('127.0.0.1', self.remote_port, True, True, 5000, 5)
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
        _, self.target_neck = vrep.simxGetObjectHandle(
            self.client_id, 'Mark_Head', vrep.simx_opmode_oneshot_wait)
        _, self.target_back = vrep.simxGetObjectHandle(
            self.client_id, 'Mark_Back', vrep.simx_opmode_oneshot_wait)
        _, self.target_leftfoot = vrep.simxGetObjectHandle(
            self.client_id, 'Mark_LeftFoot', vrep.simx_opmode_oneshot_wait)
        _, self.target_rightfoot = vrep.simxGetObjectHandle(
            self.client_id, 'Mark_RightFoot', vrep.simx_opmode_oneshot_wait)

        # start the simulation, in blocking mode
        vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)

        # enable streaming of state values and the observation image
        _, _, self.image = vrep.simxGetVisionSensorImage(
            self.client_id, self.camera_handle, 0, vrep.simx_opmode_streaming)
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

        _, self.target_neck_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_neck, self.camera_handle, vrep.simx_opmode_streaming)
        _, self.target_back_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_back, self.camera_handle, vrep.simx_opmode_streaming)
        _, self.target_leftfoot_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_leftfoot, self.camera_handle, vrep.simx_opmode_streaming)
        _, self.target_rightfoot_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_rightfoot, self.camera_handle, vrep.simx_opmode_streaming)

        self.total_reward = 0
        self.episode_reward = 0
        self.episode_step = 0
        self.max_episode_step = DEFAULT_MAX_EPISODE_STEP

    def begin_episode(self, max_episode_step=DEFAULT_MAX_EPISODE_STEP):
        self.max_episode_step = max_episode_step
        if self.episode_step >= self.max_episode_step or self.episode_terminate():
            vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)
            self.start()
        self.episode_reward = 0
        self.episode_step = 0

        # trigger several simulation step for initialization
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxSynchronousTrigger(self.client_id)

    def play(self, a):
        self.episode_step += 1
        reward = 0.0
        scaled_a = self.normalize_action(a)
        # scaled_a = a

        for i in xrange(self.frame_skip):
            # send control signals
            vrep.simxSetStringSignal(self.client_id, 'thrust',
                                     vrep.simxPackFloats(scaled_a), vrep.simx_opmode_oneshot)
            # trigger next simulation step
            vrep.simxSynchronousTrigger(self.client_id)
            self._read_vrep_data()
            self._get_track_coordinates()
            # get reward
            reward += self._get_reward()

        self.total_reward += reward
        # print 'Single step reward:%f' % reward
        self.episode_reward += reward
        # save current observation or next step observation
        ob = self.get_observation()
        terminate_flag = self.episode_terminate()
        # whether clip the reward or not
        self.replay_memory.append(ob, a, reward, terminate_flag)
        return reward, terminate_flag

    def episode_terminate(self):
        return self.episode_step > 0 and (
            self.quadcopter_pos[2] <= 0
            or abs(self.quadcopter_pos[2]) >= 5
            or abs(self.angular_velocity_b[0]) >= 1
            or abs(self.angular_velocity_b[1]) >= 1
            or abs(self.quadcopter_orientation[0]) >= 1
            or abs(self.quadcopter_orientation[1]) >= 1
            or abs(self.target_coordinates[0]) > 0.5
            or abs(self.target_coordinates[1]) > 0.5
                                          )

    def get_observation(self):
        return numpy.array(self.linear_velocity_b.tolist() +
                           self.angular_velocity_b +
                           self.target_coordinates.tolist(), dtype='float32')
        # return numpy.array(self.linear_velocity_b.tolist() + self.angular_velocity_b + self.quadcopter_orientation[0:2], dtype='float32')

    @property
    def state_enabled(self):
        return self.replay_memory.size >= self.replay_memory.history_length

    '''
    stacking specified history length to represent the state.
    when history length crosses the border, just use a tile of the current step.
    '''
    def current_state(self):
        # return reset states (currently all reset to zero)
        if self.episode_step == 0:
            return numpy.zeros((self.replay_memory.history_length,) + self.replay_memory.state_dim, dtype='float32')

        state = self.replay_memory.states.take(self.replay_memory.top-1, axis=0, mode='wrap')
        if self.replay_memory.size < self.history_length or \
                numpy.any(
                self.replay_memory.terminate_flags.take(
                    numpy.arange(self.replay_memory.top - self.replay_memory.history_length, self.replay_memory.top))):
            return numpy.tile(state.reshape((1,) + state.shape),
                              (self.replay_memory.history_length,) +
                              tuple(numpy.ones(state.ndim, dtype=numpy.int8)))
        else:
            return self.replay_memory.states.take(numpy.arange(
                self.replay_memory.top - self.replay_memory.history_length, self.replay_memory.top), axis=0, mode='wrap')


    def normalize_action(self, action):
        lb = self.action_bounds[0]
        ub = self.action_bounds[1]
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = numpy.clip(scaled_action, lb, ub)
        return scaled_action


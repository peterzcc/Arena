__author__ = 'flyers'

import numpy
import glob
import os

'''
convert a quaternion to a transformation matrix
'''
def quad2mat(q):
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


def read_linear_velocity(path):
    l = glob.glob(os.path.join(path, '*velocity.txt'))
    abs_velocity = numpy.loadtxt(l[0], dtype=numpy.float32)
    abs_velocity = abs_velocity[:, 0:3]
    f = glob.glob(os.path.join(path, '*quaternion.txt'))
    q = numpy.loadtxt(f[0], dtype=numpy.float32)
    body_velocity = numpy.zeros(abs_velocity.shape, dtype=numpy.float32)
    for i in xrange(abs_velocity.shape[0]):
        mat = quad2mat(q[i])
        body_velocity[i] = mat.transpose().dot(abs_velocity[i])

    return abs_velocity, body_velocity


def read_angular_velocity(path):
    l = glob.glob(os.path.join(path, '*velocity.txt'))
    abs_angular = numpy.loadtxt(l[0], dtype=numpy.float32)
    abs_angular = abs_angular[:, 3:6]
    l = glob.glob(os.path.join(path, '*gyro.txt'))
    body_angular = numpy.loadtxt(l[0], dtype=numpy.float32)

    return abs_angular, body_angular

def read_motor(path):
    l = glob.glob(os.path.join(path, '*motor.txt'))
    return numpy.loadtxt(l[0], dtype=numpy.float32)



# path = '/home/sliay/Documents/V-REP_PRO_EDU_V3_2_3_rev4_64_Linux/data/test'
# abs_velocity, body_velocity = read_linear_velocity(path)
# abs_angular, body_angular = read_angular_velocity(path)

# print body_velocity
# print abs_angular.shape, body_angular.shape

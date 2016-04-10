import mxnet as mx
import mxnet.ndarray as nd
import numpy
from arena import Base
import cv2
from arena.utils import *


def load_bb(path):
    a = numpy.loadtxt(path)
    cx = a[:, ::2].mean(axis=1)
    cy = a[:, 1::2].mean(axis=1)
    sx = a[:, ::2].max(axis=1) - a[:, ::2].min(axis=1) + 1
    sy = a[:, 1::2].max(axis=1) - a[:, 1::2].min(axis=1) + 1
    cx = cx / 480 * 2 -1
    cy = cy / 360 * 2 - 1
    sx = sx / 480 * 2 - 1
    sy = sy / 360 * 2 - 1
    return numpy.vstack((cx, cy)).T, numpy.vstack((sx, sy)).T

data = mx.symbol.Variable('data')
center = mx.symbol.Variable('center')
size = mx.symbol.Variable('size')
net = mx.symbol.SpatialGlimpse(data=data, center=center, size=size, output_shape=(224, 224), scale=1.0, name='spatial_glimpse')

bgr_img = cv2.imread('D:\\HKUST\\tracking\\vot-workshop\\sequences\\sequences\\bag\\00000001.jpg')
b, g, r = cv2.split(bgr_img)       # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb

data_npy = numpy.rollaxis(rgb_img, 2, 0).reshape((1, 3, 360, 480))

center_npy, size_npy = load_bb("D:\\HKUST\\tracking\\vot-workshop\\sequences\\sequences\\bag\\groundtruth_parsed.txt")

center_npy = center_npy[:1, :]
size_npy = size_npy[:1, :]
data_shapes = {'data': (1, 3, 360, 480),
               'center': (1, 2),
               'size': (1, 2)}

glimpse_test_net = Base(data_shapes=data_shapes, sym=net, name='GlimpseTest', ctx=mx.gpu())

out_img = glimpse_test_net.forward(batch_size=1, data=nd.array(data_npy, ctx=mx.gpu()),
                                   center=nd.array(center_npy, ctx=mx.gpu()), size=nd.array(size_npy, ctx=mx.gpu()))[0].asnumpy()
r, g, b = cv2.split(numpy.rollaxis(out_img.reshape((3, 224, 224)), 0, 3))
reshaped_img = cv2.merge([b,g,r])

cv2.imshow('image', reshaped_img/255.0)
cv2.waitKey(0)
#
# shapes = []
#
# arr = {'arg_%d' % i: mx.random.uniform(-10.0, 10.0, shape) for i, shape in
#        zip(range(len(shapes)), shapes)}
# arr_grad = {'arg_%d' % i: mx.nd.zeros(shape) for i, shape in zip(range(len(shapes)), shapes)}
#
# up = mx.sym.UpSampling(*[mx.sym.Variable('arg_%d' % i) for i in range(len(shapes))],
#                        sample_type='nearest', scale=root_scale)
# exe = up.bind(mx.cpu(), args=arr, args_grad=arr_grad)
# exe.forward(is_train=True)
# exe.backward(exe.outputs)
import mxnet as mx
import mxnet.ndarray as nd
import numpy
from arena import Base
import cv2
import time
from arena.utils import *


def load_roi(path, height=360, width=480, num=100):
    a = numpy.loadtxt(path, delimiter=',')
    cx = a[:, ::2].mean(axis=1)
    cy = a[:, 1::2].mean(axis=1)
    sx = a[:, ::2].max(axis=1) - a[:, ::2].min(axis=1) + 1
    sy = a[:, 1::2].max(axis=1) - a[:, 1::2].min(axis=1) + 1
    cx = cx / width * 2 -1
    cy = cy / height * 2 - 1
    sx = sx / width * 2 - 1
    sy = sy / height * 2 - 1
    rois = numpy.vstack((cx, cy, sx, sy)).T
    return rois[:num, :]

def load_image(path, height=360, width=480, num=100):
    data_npy = numpy.zeros((num, 3, height, width), dtype=numpy.float32)
    for i in range(num):
        image_path = path + "\\%08d.jpg" % (i+1)
        print image_path
        bgr_img = cv2.imread(image_path)
        b, g, r = cv2.split(bgr_img)  # get b,g,r
        data_npy[i, :, :, :] = numpy.rollaxis(cv2.merge([r, g, b]), 2, 0)
    return data_npy

def pyramid_glimpse(data, roi, depth, scale, output_shape, name):
    l = []
    curr_scale = 1.0
    if type(roi) is tuple:
        roi = mx.symbol.Concat(*roi, num_args=depth)
    for i in range(depth):
        l.append(mx.symbol.SpatialGlimpse(data=data, roi=roi,
                                          output_shape=output_shape,
                                          scale=curr_scale, name="%s-%d" %(name, i)))
        curr_scale *= scale
    ret = mx.symbol.Concat(*l, num_args=depth, name="%s-concat" %name)
    return ret

ctx = mx.cpu()
data = mx.symbol.Variable('data')
center = mx.symbol.Variable('center')
size = mx.symbol.Variable('size')
roi = mx.symbol.Variable('roi')
print type(data)
depth = 3
scale = 1.5
rows = 720
cols = 1280
path= "D:\\HKUST\\advanced\\vot-workshop\\sequences\\sequences\\ball1"
net = pyramid_glimpse(data=data, roi=roi, depth=depth, scale=scale, output_shape=(107, 107),
                      name='spatial_glimpse')
batch_size = 50

data_arr = nd.array(load_image(path=path,
                               num=batch_size, height=rows, width=cols), ctx=ctx)

roi_arr = nd.array(load_roi(path=path + "\\groundtruth.txt",
                            num=batch_size, height=rows, width=cols), ctx=ctx)

print data_arr.shape
print roi_arr.shape
data_shapes = {'data': (batch_size, 3, rows, cols),
               'roi': (batch_size, 4)}

glimpse_test_net = Base(data_shapes=data_shapes, sym=net, name='GlimpseTest', ctx=ctx)

start = time.time()
out_imgs = glimpse_test_net.forward(batch_size=batch_size, data=data_arr, roi=roi_arr)[0].asnumpy()
end = time.time()
print 'Time:', end-start
print out_imgs.shape
for i in range(batch_size):
    for j in range(depth):
        r, g, b = cv2.split(numpy.rollaxis(out_imgs[i, j*3:(j+1)*3], 0, 3))
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
import mxnet as mx
import mxnet.ndarray as nd
import numpy
from arena import Base
import cv2
import time
from arena.utils import *


def load_roi(path, height=360, width=480, num=100):
    a = numpy.loadtxt(path)
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
ctx = mx.cpu()
data = mx.symbol.Variable('data')
roi = mx.symbol.Variable('roi')
net = mx.symbol.SpatialGlimpse(data=data, roi=roi, output_shape=(224, 224), scale=1.0, name='spatial_glimpse')
batch_size = 180

data_arr = nd.array(load_image(path="D:\\HKUST\\tracking\\vot-workshop\\sequences\\sequences\\bag",
                               num=batch_size), ctx=ctx)

roi_arr = nd.array(load_roi("D:\\HKUST\\tracking\\vot-workshop\\sequences\\sequences\\bag\\groundtruth_parsed.txt",
                            num=batch_size), ctx=ctx)

print data_arr.shape
print roi_arr.shape
data_shapes = {'data': (batch_size, 3, 360, 480),
               'roi': (batch_size, 4)}

glimpse_test_net = Base(data_shapes=data_shapes, sym=net, name='GlimpseTest', ctx=ctx)

start = time.time()
out_imgs = glimpse_test_net.forward(batch_size=batch_size, data=data_arr, roi=roi_arr)[0].asnumpy()
end = time.time()
print 'Time:', end-start
print out_imgs.shape
for i in range(batch_size):
    r, g, b = cv2.split(numpy.rollaxis(out_imgs[i], 0, 3))
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
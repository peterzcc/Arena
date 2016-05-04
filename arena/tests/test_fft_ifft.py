import mxnet as mx
from arena import Base
import numpy
import pyfftw
import time
import mxnet.ndarray as nd
import cv2


a = mx.symbol.Variable('a')
b = mx.symbol.Variable('b')
a_fft = mx.symbol.FFT2D(data=a)
a_fft = mx.symbol.BlockGrad(data=a_fft)
a_recons = mx.symbol.IFFT2D(data=a_fft, output_shape=(64, 64))
c = mx.symbol.Flatten(a_fft)
d_pred = mx.symbol.FullyConnected(data=c, num_hidden=64)
d = mx.symbol.LinearRegressionOutput(data=d_pred, label=b)
data_shapes = {'a': (10, 96*3, 64, 64)}
a_npy = numpy.zeros((1, 1, 3, 3))
a_npy[0,0,0,:] = numpy.array([1,2,1])
a_npy[0,0,1,:] = numpy.array([2,3,2])
a_npy[0,0,2,:] = numpy.array([2,3,4])
optimizer = mx.optimizer.create(name='sgd', learning_rate=0.01,
                                clip_gradient=None,
                                rescale_grad=1.0, wd=0.)
updater = mx.optimizer.get_updater(optimizer)
net = Base(sym=a_recons, data_shapes=data_shapes, name='net', ctx=mx.gpu())


cpu_time = 0
gpu_time = 0
data = numpy.zeros((10, 96*3, 64, 64))
for i in range(100):
    data[:] = numpy.random.rand(10, 96*3, 64, 64)
    start = time.time()
    output_fftw = pyfftw.interfaces.numpy_fft.irfft2(pyfftw.interfaces.numpy_fft.rfft2(data))
    end = time.time()
    cpu_time += end-start
    start = time.time()
    output_mxnet = net.forward(data_shapes=data_shapes, a=nd.array(data, ctx=mx.gpu()))[0].asnumpy()
    end = time.time()
    gpu_time += end-start
    print numpy.square(output_mxnet - output_fftw.real).sum()
print cpu_time
print gpu_time
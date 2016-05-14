import mxnet as mx
from arena import Base
import numpy
import pyfftw
import time
import mxnet.ndarray as nd
import cv2

def test_fft():
    a = mx.symbol.Variable('a')
    b_fft = mx.symbol.Variable('b_fft')
    a = mx.symbol.FFT2D(data=a)
    c = mx.symbol.Reshape(mx.symbol.sum(mx.symbol.square(a - b_fft)), shape=(1, 1))
    c = mx.symbol.MakeLoss(c)

    data_shape = (1, 2, 16, 15)
    fft_shapes = (data_shape[0], data_shape[1], data_shape[2], 2*(data_shape[3]/2 + 1))

    a_npy = numpy.random.rand(*data_shape)
    b_npy = numpy.random.rand(*data_shape)
    b_fft_unfold_npy = numpy.empty(fft_shapes)
    b_fft_npy = pyfftw.interfaces.numpy_fft.rfft2(b_npy)
    b_fft_unfold_npy[:, :, :, ::2] = numpy.real(b_fft_npy)
    b_fft_unfold_npy[:, :, :, 1::2] = numpy.imag(b_fft_npy)
    print b_fft_npy
    print b_fft_unfold_npy
    a_grad = nd.empty(data_shape, ctx=mx.gpu())
    net = c.bind(mx.gpu(), args={'a': nd.array(a_npy, ctx=mx.gpu()), 'b_fft': nd.array(b_fft_unfold_npy, ctx=mx.gpu())},
                 args_grad={'a': a_grad})
    net.forward(is_train=True)
    base_loss = net.outputs[0].asnumpy()
    print base_loss
    net.backward()
    grad_compute_backward = a_grad.asnumpy()
    eps = 1E-3
    grad_finite_difference_cufft = numpy.empty((a_npy.size), dtype=numpy.float32)
    #grad_finite_difference_fftw = numpy.empty((a_npy.size), dtype=numpy.float32)
    for i in range(a_npy.size):
        z = numpy.zeros((a_npy.size,), dtype=numpy.float32)
        z[i] = eps
        dat = a_npy + z.reshape(a_npy.shape)
        net.arg_dict['a'][:] = dat
        net.forward(is_train=False)
        loss = net.outputs[0].asnumpy()
        loss2 = numpy.square(numpy.abs(pyfftw.interfaces.numpy_fft.rfft2(dat) - b_fft_npy)).sum()
        grad_finite_difference_cufft[i] = (loss[0] - base_loss[0])/eps
        #grad_finite_difference_fftw[i] = (loss2 - base_loss[0])/eps
    grad_finite_difference_cufft = grad_finite_difference_cufft.reshape(a_npy.shape)
    #grad_finite_difference_fftw = grad_finite_difference_fftw.reshape(a_npy.shape)
    print "Compute By Backward:", grad_compute_backward
    print "Compute By FD-CUFFT:", grad_finite_difference_cufft
    #print "Compute By FD-FFTW", grad_finite_difference_fftw
    print numpy.square(grad_finite_difference_cufft - grad_compute_backward).mean()

    lr = 0.001
    net.arg_dict['a'][:] = a_npy
    for i in range(100):
        net.forward(is_train=True)
        loss = net.outputs[0].asnumpy()
        print numpy.square(net.arg_dict['a'].asnumpy() - b_npy).sum()
        net.backward()
        net.arg_dict['a'][:] -= lr * a_grad


def test_ifft():
    a_fft = mx.symbol.Variable('a_fft')
    b = mx.symbol.Variable('b')
    a = mx.symbol.IFFT2D(data=a_fft)



def speed_compare_cufft_fftw():
    a = mx.symbol.Variable('a')
    b = mx.symbol.Variable('b')
    a_fft = mx.symbol.FFT2D(data=a, batchsize=32)
    a_recons = mx.symbol.IFFT2D(data=a_fft, output_shape=(64, 64), batchsize=32)
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
        output_mxnet = net.forward(is_train=False, data_shapes=data_shapes, a=nd.array(data, ctx=mx.gpu()))[0].asnumpy()
        end = time.time()
        gpu_time += end-start
        print numpy.square(output_mxnet - output_fftw.real).sum()
    print cpu_time
    print gpu_time
test_fft()
#speed_compare_cufft_fftw()
import mxnet as mx
from arena import Base
import numpy
import pyfftw
import time
import mxnet.ndarray as nd
import cv2

def foo(A, B):
    fft_A = pyfftw.interfaces.numpy_fft.rfft2(A)
    fft_B = pyfftw.interfaces.numpy_fft.rfft2(B)
    ret = 0.5 * numpy.square(numpy.abs(fft_A - fft_B)).sum()
    return ret


def grad_foo(A, B):
    fft_A = pyfftw.interfaces.numpy_fft.rfft2(A)
    fft_B = pyfftw.interfaces.numpy_fft.rfft2(B)
    print 'rfft_A =', fft_A
    print 'fft_A =', pyfftw.interfaces.numpy_fft.fft2(A)
    print 'rfft_B =', fft_B
    print pyfftw.interfaces.numpy_fft.irfft2(fft_A - fft_A, s=A.shape)
    diff = fft_A - fft_B
    print 'diff =', diff

    diff[:] *= fft_A.shape[0] * fft_A.shape[1]
    #diff[0] *= A.shape[1]/2
    grad = pyfftw.interfaces.numpy_fft.irfft2(diff, s=A.shape)
    return grad

A = numpy.random.rand(4, 4)
B = numpy.random.rand(4, 4)

eps = 1E-10
grad_finite_difference = numpy.zeros((4, 4), dtype=numpy.float32)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        Z = numpy.zeros((4, 4), dtype=numpy.float32)
        Z[i, j] = eps
        grad_finite_difference[i, j] = (foo(A+Z, B) - foo(A, B))/eps
print grad_finite_difference
grad_compute = grad_foo(A, B)
print grad_compute
print numpy.square(grad_compute - grad_finite_difference).sum()
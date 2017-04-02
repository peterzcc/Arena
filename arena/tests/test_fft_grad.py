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
    print fft_A[:, 1:]
    print 'fft_A =', pyfftw.interfaces.numpy_fft.fft2(A)
    print 'rfft_B =', fft_B
    print pyfftw.interfaces.numpy_fft.irfft2(fft_A - fft_A, s=A.shape)
    diff = fft_A - fft_B
    print 'diff =', diff

    diff[:] *= A.shape[0] * A.shape[1]
    if 0 == A.shape[1] % 2:
        diff[:, 1:-1] /= 2
    else:
        diff[:, 1:] /= 2
    grad = pyfftw.interfaces.numpy_fft.irfft2(diff, s=A.shape)
    return grad

def foo_irfft(fft_A, fft_B):
    A = pyfftw.interfaces.numpy_fft.irfft2(fft_A)
    B = pyfftw.interfaces.numpy_fft.irfft2(fft_B)
    ret = 0.5 * numpy.square(numpy.abs(A - 2*B)).sum()
    return ret

def grad_foo_irfft(fft_A, fft_B):
    print 'rfft_A =', fft_A
    print 'rfft_B =', fft_B
    A = pyfftw.interfaces.numpy_fft.irfft2(fft_A)
    B = pyfftw.interfaces.numpy_fft.irfft2(fft_B)
    print 'A =', A
    print 'B =', B
    print fft_A[:, 1:]
    print 'fft_A =', pyfftw.interfaces.numpy_fft.fft2(A)
    print pyfftw.interfaces.numpy_fft.irfft2(fft_A - fft_A, s=A.shape)
    diff = A - 2*B
    print 'diff =', diff
    grad = pyfftw.interfaces.numpy_fft.rfft2(diff, s=A.shape)
    print 'grad =', grad
    grad[:] /= A.shape[0] * A.shape[1]
    if 0 == A.shape[1] % 2:
        grad[:, 1:-1] *= 2
    else:
        grad[:, 1:] *= 2

    return grad

def test_rfft2d_grad():
    in_shape = (64, 64)
    A = numpy.random.rand(*in_shape)*10
    B = numpy.random.rand(*in_shape)*10

    eps = 1E-6
    grad_finite_difference = numpy.zeros(in_shape, dtype=numpy.float32)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Z = numpy.zeros(in_shape, dtype=numpy.float32)
            Z[i, j] = eps
            grad_finite_difference[i, j] = (foo(A+Z, B) - foo(A, B))/eps

    grad_compute = grad_foo(A, B)
    print 'grad_finite_difference:', grad_finite_difference
    print 'grad_compute:', grad_compute
    print grad_compute/grad_finite_difference
    print numpy.square(grad_compute - grad_finite_difference).mean()

def test_irfft2d_grad():
    in_shape = (5, 5)
    A = numpy.random.rand(*in_shape)*10
    B = numpy.random.rand(*in_shape)*10
    fft_A = pyfftw.interfaces.numpy_fft.rfft2(A)
    fft_B = pyfftw.interfaces.numpy_fft.rfft2(B)
    eps = 1E-3
    grad_finite_difference = numpy.zeros(fft_A.shape, dtype=numpy.complex)
    for i in range(fft_A.shape[0]):
        for j in range(fft_A.shape[1]):
            Z = numpy.zeros(fft_A.shape, dtype=numpy.complex)
            Z[i, j] = eps + 0j
            grad_finite_difference[i, j] += (foo_irfft(fft_A+Z, fft_B) - foo_irfft(fft_A, fft_B))/eps
            Z[i, j] = 0 + 1j*eps
            grad_finite_difference[i, j] += (foo_irfft(fft_A + Z, fft_B) - foo_irfft(fft_A, fft_B)) / eps * 1j

    grad_compute = grad_foo_irfft(fft_A, fft_B)
    print 'grad_finite_difference:', grad_finite_difference
    print 'grad_compute:', grad_compute
    print grad_compute/grad_finite_difference
    print numpy.square(grad_compute - grad_finite_difference).mean()

test_irfft2d_grad()
import numpy
import scipy
from scipy.stats import entropy

def softmax(x):
    y = numpy.exp(x - x.max())
    y /= y.sum()
    return y

def softmax_entropy(x):
    y = softmax(x)
    return -entropy(y)

def softmax_entropy_grad(x):
    y = softmax(x)
    grad = y * entropy(y) + numpy.nan_to_num(y * numpy.log(y))
    return grad

siz = 128
a = numpy.random.rand(siz)
grad_finite_difference = numpy.zeros(siz)
for i in range(siz):
    eps = numpy.zeros(siz, dtype=numpy.float32)
    eps[i] = 1E-10
    grad_finite_difference[i] = (softmax_entropy(a+eps) - softmax_entropy(a)) / (1E-10)
grad_compute = softmax_entropy_grad(a)
print numpy.square(grad_finite_difference - grad_compute).sum()
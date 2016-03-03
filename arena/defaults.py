__author__ = 'sxjscience'
import mxnet as mx
import numpy.random
import os
_ctx = mx.cpu()
_numpy_rng = numpy.random.RandomState(123456)
def get_ctx():
    return _ctx

def get_numpy_rng():
    return _numpy_rng
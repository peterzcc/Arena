import mxnet as mx
import mxnet.ndarray as nd
import numpy
a = mx.random.normal(0, 1, shape=(2,2))
print nd.norm(a).asnumpy()
print numpy.sqrt((a.asnumpy()**2).sum())

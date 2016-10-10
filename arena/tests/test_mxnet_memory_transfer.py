import mxnet as mx
import mxnet.ndarray as nd
import time
N = 10
a = mx.random.normal(0, 1, (4, 1024, 1024), ctx=mx.gpu(0))
b = mx.nd.empty((4, 1024, 1024), ctx=mx.cpu())
c = mx.nd.empty((4, 1024, 1024), ctx=mx.gpu())
nd.waitall()
a.asnumpy()
start = time.time()
for i in range(N):
    b[:] = a
    nd.waitall()
avg_time = (time.time() - start)/N
print('GPU->CPU, GB/s: %g' %(4 * a.size * 1E-9 / avg_time))

start = time.time()
for i in range(N):
    a[:] = b
    nd.waitall()
avg_time = (time.time() - start)/N
print('CPU->GPU, GB/s: %g' %(4 * a.size * 1E-9 / avg_time))

start = time.time()
for i in range(N):
    c[:] = a
    nd.waitall()
avg_time = (time.time() - start)/N
print('GPU->GPU, GB/s: %g' %(4 * a.size * 1E-9 / avg_time))
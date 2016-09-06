import mxnet as mx
import mxnet.ndarray as nd
import time
N = 10
a = mx.random.normal(0, 1, (4, 1024, 1024), ctx=mx.gpu(0))
nd.waitall()
a.asnumpy()
start = time.time()
for i in range(N):
    a.asnumpy()
avg_time = (time.time() - start)/N
print('GB/s: %g' %(4 * a.size * 1E-9 / avg_time))
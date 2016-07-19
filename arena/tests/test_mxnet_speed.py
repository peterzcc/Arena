import mxnet as mx
import mxnet.ndarray as nd
import numpy
import time

ww = nd.ones((10,), ctx=mx.gpu()).asnumpy()
time_npy = 0
time_mxcpu = 0
time_npymxcpu = 0

temp = nd.empty((32,4,84,84),ctx=mx.cpu())
for i in range(100):
    arr_npy = numpy.random.normal(0, 1, (32, 4, 84, 84))
    arr_mxcpu = mx.random.normal(0, 1, (32, 4, 84, 84), ctx=mx.cpu())
    arr_mxcpu.asnumpy()

    start = time.time()
    arr_gpu = nd.array(arr_npy, ctx=mx.gpu())
    arr_gpu.wait_to_read()
    end = time.time()
    print "Numpy CPU copying time:", end-start
    time_npy += end-start


    start = time.time()
    arr_gpu1 = arr_mxcpu.copyto(mx.gpu())
    arr_gpu1.wait_to_read()
    end = time.time()
    print "MXNet CPU copying time:", end-start
    time_mxcpu += end-start

    start = time.time()
    temp._sync_copyfrom(arr_npy)
    arr_gpu2 = temp.copyto(mx.gpu())
    arr_gpu2.wait_to_read()
    end = time.time()
    time_npymxcpu += end - start
    print "Npy MXNet CPU copying time:", end-start

print "MXCpu:%f, Npy:%f, NpyMXCPU:%f" %(time_mxcpu, time_npy, time_npymxcpu)
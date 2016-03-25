import mxnet as mx
import mxnet.ndarray as nd
import mxnet.kvstore as kvs

kv_store_typ = 'dist_sync'
ctx = mx.gpu(0)
kv = kvs.create(kv_store_typ)
shape = (2,3)
kv.init(3, nd.ones(shape, ctx=ctx))
a = mx.nd.zeros(shape, ctx=ctx)
kv.pull(3, out = a)
print a.asnumpy()
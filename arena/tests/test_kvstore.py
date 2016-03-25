import mxnet as mx
import mxnet.ndarray as nd
import mxnet.kvstore as kvs

def update(key, input, stored):
    print "update on key: %d" % key
    stored += input * 2

kv_store_typ = 'dist_sync'
ctx = mx.gpu(0)
kv = kvs.create(kv_store_typ)
shape = (2,3)
kv.init(3, nd.ones(shape, ctx=ctx))
a = mx.nd.zeros(shape, ctx=ctx)
kv.pull(3, out = a)
print a.asnumpy()
kv.push(3, mx.nd.ones(shape)*8)
kv.pull(3, out = a)
print a.asnumpy()

kv._set_updater(update)
kv.pull(3, out=a)
print a.asnumpy()
kv.push(3, mx.nd.ones(shape))
kv.pull(3, out=a)
print a.asnumpy()
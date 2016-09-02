import mxnet as mx
import numpy as np
"""
data = mx.nd.array(range(6)).reshape((2,3))
#print "input shape = %s" % data.shape
#print "data = %s" % (data.asnumpy(), )
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
for dim in range(2):
    cat = mx.sym.Concat(a, b, dim=dim)
    exe = cat.bind(ctx=mx.cpu(), args={'a':data, 'b':data})
    exe.forward()
    out = exe.outputs[0]
    print "concat at dim = %d" % dim
    print "shape = %s" % (out.shape, )
    print "results = %s" % (out.asnumpy(), )
"""
num_of_instance = 1
a = mx.symbol.Variable('a')
b = mx.symbol.Variable('b')
c = mx.symbol.softmax_cross_entropy(a,b)
exe = c.simple_bind(ctx=mx.cpu(), a=(num_of_instance,3), b=(num_of_instance,))
print exe.arg_dict

a_npy = np.random.rand(num_of_instance,3)
print "a",a_npy
b_npy = np.random.randint(0,3,num_of_instance)
print "b",b_npy
exe.arg_dict['a'][:] = a_npy
exe.arg_dict['b'][:] = b_npy
exe.forward(is_train=False)

print exe.outputs[0].asnumpy()
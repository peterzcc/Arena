import mxnet as mx
import numpy as np
from mxnet.test_utils import *

a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = mx.sym.batch_cconv(a, b)
a_npy = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
b_npy = np.array([[1, 2, 3], [2, 3, 4]])

exe = c.simple_bind(ctx=mx.gpu(), a=a_npy.shape, b=b_npy.shape)
outputs = exe.forward(is_train=True, a=a_npy, b=b_npy)
outputs[0].wait_to_read()
print(outputs[0].asnumpy())

check_numeric_gradient(sym=c, ctx=mx.gpu(),
                       location=(a_npy, b_npy),
                       numeric_eps=1E-3,
                       check_eps=0.01)

data = mx.symbol.Variable(name='data')
rois = mx.symbol.Variable(name='rois')
test = mx.symbol.ROIPooling(data=data, rois=rois, pooled_size=(6, 6), spatial_scale=1)

x1 = np.random.normal(size=(4, 3, 12, 8))
x2 = np.array([[0, 1, 1, 6, 6], [2, 6, 2, 7, 11], [1, 3, 1, 5, 10], [0, 3, 3, 3, 3]])

check_numeric_gradient(sym=test, location=[x1, x2], ctx=mx.cpu(),
                       grad_nodes={'data': 'add', 'rois': 'null'},
                       numeric_eps=1e-3, check_eps=1e-2)
check_numeric_gradient(sym=test, location=[x1, x2], ctx=mx.cpu(),
                       grad_nodes={'data': 'add', 'rois': 'write'},
                       numeric_eps=1e-3, check_eps=1e-2)
check_numeric_gradient(sym=test, location=[x1, x2], ctx=mx.cpu(),
                       grad_nodes={'data': 'write', 'rois': 'write'},
                       numeric_eps=1e-3, check_eps=1e-2)

sym_in = mx.sym.Variable('sym_in')
sym_out = mx.sym.sum(sym_in, axis=(1, 3))
# for i in range(2):
#     check_numeric_gradient(sym=sym_out, ctx=mx.gpu(),
#                            location=(np.random.normal(0, 2, (10, 10, 10)),),
#                            check_eps=0.05)
print(check_speed(sym=sym_out, ctx=mx.gpu(), sym_in=(20, 20, 20, 20), typ="whole"))
print(check_speed(sym=sym_out, ctx=mx.cpu(), sym_in=(20, 20, 20, 20), typ="whole"))
print(check_speed(sym=sym_out, ctx=mx.gpu(), sym_in=(20, 20, 20, 20), typ="forward"))
print(check_speed(sym=sym_out, ctx=mx.cpu(), sym_in=(20, 20, 20, 20), typ="forward"))
tic = time.time()
a = np.empty((20, 20))
b = np.random.normal(0, 1, (20, 20, 20, 20))
for i in range(100):
    a[:] = b.sum(axis=(1, 3))
toc = time.time()
print((toc - tic) * 1.0 / 100)

print('Begin Benchmarking batch_cconv')
print(check_speed(sym=c, ctx=mx.gpu(), a=(2048, 100), b=(2048, 3)))
print(check_speed(sym=c, ctx=mx.cpu(), a=(2048, 100), b=(2048, 3)))
print(check_speed(sym=c, ctx=mx.gpu(), a=(2048, 50), b=(2048, 5)))
print(check_speed(sym=c, ctx=mx.cpu(), a=(2048, 50), b=(2048, 5)))
print(check_speed(sym=c, ctx=mx.gpu(), a=(2048, 20), b=(2048, 3)))
print(check_speed(sym=c, ctx=mx.cpu(), a=(2048, 20), b=(2048, 3)))

# a = mx.sym.Variable('a')
# b = mx.sym.Concat(a, a, num_args=2, dim=0)
# check_numeric_gradient(sym=b, ctx=mx.gpu(),
#                        location=(np.random.normal(0, 1, (10, 10, 10)),),
#                        numeric_eps=1E-5,
#                        check_eps=0.01)

# a = mx.sym.Variable('a')
# b = mx.sym.transpose(a)
# print('Begin Benchmarking transpose')
# print(check_forward_speed(sym=b, ctx=mx.gpu(), a=(100000, 128)))
# print(check_forward_speed(sym=b, ctx=mx.gpu(), a=(100000, 512)))
# print(check_forward_speed(sym=b, ctx=mx.gpu(), a=(500000, 1024)))


data = mx.sym.Variable('data')
embedding_weight = mx.sym.Variable('embedding_weight')
embed = mx.sym.Embedding(data=data, weight=embedding_weight, input_dim=100000, output_dim=150)
print('Begin Benchmarking embedding')
print(check_speed(sym=embed, ctx=mx.gpu(),
                  location={'data': np.random.randint(0, 100000, (128, 100)),
                            'embedding_weight': np.random.normal(0, 1, (100000, 150))},
                  grad_req={'data': 'null', 'embedding_weight': 'add'},
                  typ="whole"))
print(check_speed(sym=embed, ctx=mx.cpu(),
                  location={'data': np.random.randint(0, 100000, (128, 100)),
                            'embedding_weight': np.random.normal(0, 1, (100000, 150))},
                  grad_req={'data': 'null', 'embedding_weight': 'add'},
                  typ="whole"))
print(check_speed(sym=embed, ctx=mx.gpu(),
                  location={'data': np.random.randint(0, 100000, (128, 100)),
                            'embedding_weight': np.random.normal(0, 1, (100000, 150))},
                  typ="forward"))
print(check_speed(sym=embed, ctx=mx.cpu(),
                  location={'data': np.random.randint(0, 100000, (128, 100)),
                            'embedding_weight': np.random.normal(0, 1, (100000, 150))},
                  typ="forward"))

a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = mx.sym.batch_dot(a, b)
d = mx.sym.broadcast_mul(a, b)
d = mx.sym.sum(d, axis=2, keepdims=True)
print('Begin Benchmarking batch_dot')
print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 20, 100), b=(128, 100, 1)))
print(check_speed(sym=c, ctx=mx.cpu(), a=(128, 20, 100), b=(128, 100, 1)))
print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 100, 128), b=(128, 128, 1)))
print(check_speed(sym=c, ctx=mx.cpu(), a=(128, 100, 128), b=(128, 128, 1)))
print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 100, 500), b=(128, 500, 1)))
print(check_speed(sym=c, ctx=mx.cpu(), a=(128, 100, 500), b=(128, 500, 1)))

print('Begin Comparing batch_dot Versus broadcast + mul')
print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 20, 100), b=(128, 100, 1)))
print(check_speed(sym=d, ctx=mx.gpu(), a=(128, 20, 100), b=(128, 1, 100)))
print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 100, 128), b=(128, 128, 1)))
print(check_speed(sym=d, ctx=mx.gpu(), a=(128, 100, 128), b=(128, 1, 128)))
print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 200, 500), b=(128, 500, 1)))
print(check_speed(sym=d, ctx=mx.gpu(), a=(128, 200, 500), b=(128, 1, 500)))
# print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 200, 1), b=(128, 1, 100)))
# print(check_speed(sym=c, ctx=mx.cpu(), a=(128, 200, 1), b=(128, 1, 100)))
# print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 200, 100), b=(128, 100, 100)))
# print(check_speed(sym=c, ctx=mx.cpu(), a=(128, 200, 100), b=(128, 100, 100)))
# print(check_speed(sym=c, ctx=mx.gpu(), a=(16, 200, 100), b=(16, 100, 100)))
# print(check_speed(sym=c, ctx=mx.cpu(), a=(16, 200, 100), b=(16, 100, 100)))
#

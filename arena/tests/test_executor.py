from ..utils import ExecutorBatchSizePool
import mxnet as mx
import mxnet.ndarray as nd
import numpy

net = mx.symbol.Variable('data')
net = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=100)
net = mx.symbol.FullyConnected(data=net, name='fc2', num_hidden=100)
net = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=100)
net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=10)
net = mx.symbol.LinearRegressionOutput(data=net, name='out')

minibatch_size = 32
input_dim = 10
data_shapes = {'data': (minibatch_size, input_dim),
               'out_label': (minibatch_size, 10)}

ctx = mx.gpu()

arg_names = net.list_arguments()
aux_names = net.list_auxiliary_states()
param_names = list(set(arg_names) - set(data_shapes.keys()))
arg_shapes, output_shapes, aux_shapes = net.infer_shape(**data_shapes)
arg_name_shape = {k: s for k, s in zip(arg_names, arg_shapes)}
params = {n: nd.ones(arg_name_shape[n], ctx=ctx)*0.0001 for n in param_names}
params_grad = {n: nd.empty(arg_name_shape[n], ctx=ctx) for n in param_names}
aux_states = {k: nd.empty(s, ctx=ctx) for k, s in zip(aux_names, aux_shapes)}

exe_pool = ExecutorBatchSizePool(ctx=ctx, sym=net, data_shapes=data_shapes,
                                 params=params, params_grad=params_grad, aux_states=aux_states)


new_exe_pool = ExecutorBatchSizePool(ctx=ctx, sym=net, data_shapes=data_shapes,
                                 params=params, params_grad=params_grad, aux_states=aux_states)

print exe_pool.get(32).arg_dict['data']
print new_exe_pool.get(32).arg_dict['data']

optimizer_params = {'name': 'adagrad', 'learning_rate': 0.1, 'eps': 0.01,
                    'rescale_grad': 1.0,
                    'wd': 0}
optimizer = mx.optimizer.create(**optimizer_params)
updater = mx.optimizer.get_updater(optimizer)
data0 = nd.array(numpy.ones((32, 10)), ctx=ctx)
data0[0:10] = -2
data1 = nd.array(numpy.ones((32, 10))*2, ctx=ctx)
data1[0:10]=-1
data3 = nd.array(numpy.ones((1, 10))*300, ctx=ctx)


exe = exe_pool.get(32)

exe.arg_dict['data'][:] = data0
exe.arg_dict['out_label'][:] = data1
exe.forward(is_train=True)
output1 = exe.outputs[0]
output1.wait_to_read()
c = data0 - output1
exe.backward()
for k in params:
    updater(index=k, grad=params_grad[k], weight=params[k])
d = None
c = None
output1 = None
output2 = None
for i in range(1000):
    exe2 = exe_pool.get(1)
    # for k, v in params_grad.items():
    #     print k, v.asnumpy()
    exe2.arg_dict['data'][:] = data3
    #for v in params.values():
    #    v.wait_to_read()
    exe2.forward(is_train=False)
    output2 = exe2.outputs[0]
    output2.wait_to_read()
    d = data3 - output2


    exe.arg_dict['data'][:] = data0
    exe.arg_dict['out_label'][:] = data1
    exe.forward(is_train=True)
    output1 = exe.outputs[0]
    output1.wait_to_read()
    c = data0 - output1
    exe.backward()
    for k in params:
        updater(index=k, grad=params_grad[k], weight=params[k])


    exe2 = exe_pool.get(1)
    # for k, v in params_grad.items():
    #     print k, v.asnumpy()
    exe2.arg_dict['data'][:] = data3
    #for v in params.values():
    #    v.wait_to_read()
    exe2.forward(is_train=False)
    output2 = exe2.outputs[0]
    output2.wait_to_read()
    d = data3 - output2
print c.asnumpy()
print d.asnumpy()
print output1.asnumpy()
print output2.asnumpy()


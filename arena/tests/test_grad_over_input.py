import mxnet as mx
import mxnet.ndarray as nd
import numpy
from ..utils import ExecutorBatchSizePool


input_num = 2
hidden_num = 2
minibatch_size = 2
data = mx.symbol.Variable('data')
reward = mx.symbol.Variable('reward')
net = mx.symbol.FullyConnected(data=data, name='fc', num_hidden=hidden_num)
net = mx.symbol.LinearRegressionOutput(data=net, name='out', label=reward)
ctx = mx.gpu()
initializer=mx.init.Uniform(0.07)

data_shapes = {'data': (minibatch_size, input_num),
               'reward':(minibatch_size, hidden_num)}

arg_names = net.list_arguments()
aux_names = net.list_auxiliary_states()
arg_shapes, output_shapes, aux_shapes = net.infer_shape(**data_shapes)
arg_name_shape = {k: s for k, s in zip(arg_names, arg_shapes)}
param_names = list(set(arg_names) - set(data_shapes.keys()))
params = {n: nd.empty(arg_name_shape[n], ctx=ctx) for n in param_names}
params_grad = {n: nd.empty(arg_name_shape[n], ctx=ctx) for n in param_names}
data_grad = {n: nd.empty(data_shapes[n], ctx=ctx) for n in data_shapes}

aux_states = {k: nd.empty(s, ctx=ctx) for k, s in zip(aux_names, aux_shapes)}
for k, v in params.items():
    initializer(k, v)
    if k == 'fc_weight':
        v[:] = numpy.asarray([[1, 1], [1, 1]])

executor_pool = ExecutorBatchSizePool(ctx=ctx, sym=net,
                                      data_shapes=data_shapes,
                                      params=params, params_grad=params_grad,
                                      aux_states=aux_states)
d = numpy.asarray([[1.5, 2.5], [1.5, 2.5]])
r = numpy.asarray([[3, 4], [3, 4]])
exe = executor_pool.get(minibatch_size)
inputs_grad = executor_pool.inputs_grad_dict[minibatch_size]
for k, v in params.items():
    print k, v.asnumpy()
exe.arg_dict['data'][:] = d
exe.arg_dict['reward'][:] = r
exe.forward(is_train=True)
print exe.outputs[0].asnumpy()
exe.backward()
for k, v in inputs_grad.items():
    print k, v.asnumpy()
    print k, exe.arg_dict[k].asnumpy()
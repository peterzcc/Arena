import mxnet as mx
import mxnet.ndarray as nd
import numpy
from arena import Base
from arena.utils import *
import matplotlib.pyplot as plt


'''
Usage:
This output a normal policy
'''
class LogNormalPolicyOut(mx.operator.NDArrayOp):
    def __init__(self):
        super(LogNormalPolicyOut, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['mean', 'var', 'score']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        mean_shape = in_shape[0]
        var_shape = in_shape[1]
        score_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [mean_shape, var_shape, score_shape], \
                   [output_shape]

    def forward(self, in_data, out_data):
        mean = in_data[0]
        var = in_data[1]
        if var.size == 1:
            out_data[0][:] = mean
        else:
            out_data[0][:] = nd.sqrt(var) * mx.random.normal(0, 1, mean.shape, ctx=mean.context) \
                             + mean

    def backward(self, out_grad, in_data, out_data, in_grad):
        mean = in_data[0]
        var = in_data[1]
        action = out_data[0]
        score = in_data[2]
        if 1 == var.ndim :
            grad_mu = in_grad[0]
            grad_mu[:] = - (action - mean) * score.reshape((score.shape[0], 1)) / var
        else:
            grad_mu = in_grad[0]
            grad_var = in_grad[1]
            grad_mu[:] = - (action - mean) * score.reshape((score.shape[0], 1)) / var
            grad_var[:] = - nd.square(action - mean) * score.reshape((score.shape[0], 1)) \
                          / nd.square(var) / 2


class LogNormalPolicyOutNpy(mx.operator.NumpyOp):
    def __init__(self, rng=get_numpy_rng()):
        super(LogNormalPolicyOutNpy, self).__init__(need_top_grad=False)
        self.rng = rng

    def list_arguments(self):
        return ['mean', 'var', 'score']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        mean_shape = in_shape[0]
        var_shape = in_shape[1]
        score_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [mean_shape, var_shape, score_shape], \
                   [output_shape]

    def forward(self, in_data, out_data):
        mean = in_data[0]
        var = in_data[1]
        if 1 == var.ndim:
            out_data[0][:] = numpy.sqrt(var.reshape((var.shape[0], 1))) \
                             * self.rng.randn(*mean.shape) + mean
        else:
            out_data[0][:] = numpy.sqrt(var) * self.rng.randn(*mean.shape) + mean

    def backward(self, out_grad, in_data, out_data, in_grad):
        mean = in_data[0]
        var = in_data[1]
        action = out_data[0]
        score = in_data[2]
        if 1 == var.ndim :
            grad_mu = in_grad[0]
            grad_mu[:] = - (action - mean) * score.reshape((score.shape[0], 1)) / \
                         var.reshape((var.shape[0], 1))
        else:
            grad_mu = in_grad[0]
            grad_var = in_grad[1]
            grad_mu[:] = - (action - mean) * score.reshape((score.shape[0], 1)) / var
            grad_var[:] = - numpy.square(action - mean) * score.reshape((score.shape[0], 1))\
                          / numpy.square(var) / 2

class LogSoftmaxPolicyOut(mx.operator.NumpyOp):
    def __init__(self):
        super(LogSoftmaxPolicyOut, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'score']


'''
Usage: When the Policy is a parameteric form like \pi(a|s) = softmax(.)

softmax(a) * (scores - (softmax(a) * scores).sum(axis=1))
grad[:] = softmax(a) * scores
grad[:] -= softmax(a) * grad.sum(axis=1, keepdims=True)

'''
class SoftmaxPolicyOut(mx.operator.NumpyOp):
    def __init__(self):
        super(SoftmaxPolicyOut, self).__init__(need_top_grad=False)

class ParticlePolicyOut(mx.operator.NumpyOp):
    def __init__(self):
        super(ParticlePolicyOut, self).__init__(need_top_grad=False)

def policy_sym(action_num, output_op):
    net = mx.symbol.Variable('data')
    net = mx.symbol.FullyConnected(data=net, name='fc', num_hidden=action_num)
    net = output_op(data=net, name='dqn')
    return net


def simple_game(data, action):
    return (numpy.square(action - data*data).sum(axis=1) < 2)*1000 + \
           (numpy.square(action - data*data).sum(axis=1) < 10)*100 + 1

def update_line(hl, fig, ax, new_x, new_y):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_x))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_y))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


output_op = LogNormalPolicyOutNpy()
var = mx.symbol.Variable('var')
data = mx.symbol.Variable('data')
net_mean = mx.symbol.FullyConnected(data=data, name='fc_mean_1', num_hidden=20)
net_mean = mx.symbol.Activation(data=net_mean, name='fc_mean_relu_1', act_type='relu')
net_mean = mx.symbol.FullyConnected(data=data, name='fc_mean_2', num_hidden=20)
net_mean = mx.symbol.Activation(data=net_mean, name='fc_mean_relu_2', act_type='relu')
net_mean = mx.symbol.FullyConnected(data=net_mean, name='fc_mean_3', num_hidden=10)
net_var = mx.symbol.FullyConnected(data=data, name='fc_var_1', num_hidden=10)
net_var = mx.symbol.Activation(data=net_var, name='fc_var_softplus_1', act_type='softrelu')
net = output_op(mean=net_mean, var=net_var, name='policy')
ctx = mx.gpu()
minibatch_size = 100
data_shapes = {'data': (minibatch_size, 10), 'policy_score': (minibatch_size,)} #, 'var':(minibatch_size,)}
qnet = Base(data_shapes=data_shapes, sym=net, name='PolicyNet',
            initializer=mx.initializer.Xavier(factor_type="in", magnitude=1.0),
            ctx=ctx)
print qnet.internal_sym_names

lr = 0.00001
optimizer = mx.optimizer.create(name='sgd', learning_rate=0.00001,
                                clip_gradient=None,
                                rescale_grad=1.0, wd=0.)
updater = mx.optimizer.get_updater(optimizer)
total_iter = 1000000
stats = numpy.zeros((total_iter, 3), dtype=numpy.float32)
plt.ion()
fig, ax = plt.subplots()
lines, = ax.plot([], [])
ax.set_autoscaley_on(True)
baseline = 0
for i in range(total_iter):
#    for k, v in qnet.params.items():
#        print k, v.asnumpy()
    data = numpy.random.randn(minibatch_size, 10)
    means = qnet.forward(batch_size=minibatch_size, sym_name="fc_mean_3_output", data=data)[0].asnumpy()
    vars = qnet.forward(batch_size=minibatch_size, sym_name="fc_var_softplus_1_output", data=data)[0].asnumpy()

    outputs = qnet.forward(batch_size=minibatch_size, is_train=True, data=data) #, var=0.5*numpy.ones((minibatch_size, )))
    action = outputs[0].asnumpy()
    score = simple_game(data, action)
    baseline = baseline - 0.001 * (baseline - score.mean())
    print 'score=', score.mean(), 'err=', numpy.square(means - data*data).mean(), 'var=', vars.mean(), 'baseline=', baseline
    stats[i] = [score.mean(), numpy.square(means - data*data).mean(), vars.mean()]
    qnet.backward(batch_size=minibatch_size, policy_score=score-baseline)
    qnet.update(updater)
    update_line(lines, fig, ax, i, numpy.square(means - data*data).mean())


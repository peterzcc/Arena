import mxnet as mx
import mxnet.ndarray as nd
import numpy
from arena import Base


#TODO Normal Exploration Policy
class LogNormalPolicyOut(mx.operator.NumpyOp):
    def __init__(self, std=0.11):
        super(LogNormalPolicyOut, self).__init__(need_top_grad=False)
        self.std = std

    def list_arguments(self):
        return ['data', 'action', 'score']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = in_shape[0]
        score_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, score_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self, out_grad, in_data, out_data, in_grad):
        mean = in_data[0]
        action = in_data[1]
        score = in_data[2]
        dx = in_grad[0]
        dx[:] = (action - mean) * score.reshape((score.shape[0], 1)) /(self.std*self.std)

class LogSoftmaxPolicyOut(mx.operator.NumpyOp):
    

def policy_sym(action_num, output_op):
    net = mx.symbol.Variable('data')
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=action_num)
    net = output_op(data=net, name='dqn')
    return net
import mxnet as mx
import mxnet.ndarray as nd
import numpy
from utils import *



# TODO NDArrayOP will cause some troubles see `https://github.com/dmlc/mxnet/issues/1720'
class DQNOutputOp(mx.operator.NDArrayOp):
    def __init__(self):
        super(DQNOutputOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        reward_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, reward_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = out_data[0]
        action = in_data[1]
        reward = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        dx[:] = nd.fill_element_0index(dx,
                                       nd.clip(nd.choose_element_0index(x, action) - reward, -1, 1),
                                       action)


# TODO Regression Output has none differential for label, we may need to fix that
class DQNOutputNpyOp(mx.operator.NumpyOp):
    def __init__(self):
        super(DQNOutputNpyOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        reward_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, reward_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = out_data[0]
        action = in_data[1].astype(numpy.int)
        reward = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        dx[numpy.arange(action.shape[0]), action] \
            = numpy.clip(x[numpy.arange(action.shape[0]), action] - reward, -1, 1)



'''
Name: LogNormalPolicy
Usage: This OP outputs actions generated by a policy with normal distribution.
       The loss function for backward operation is set to be \sum_i - log(N(a_i|m_i, v_i)) * R_i
'''


class LogNormalPolicy(mx.operator.NumpyOp):
    def __init__(self, rng=get_numpy_rng(), deterministic=False):
        super(LogNormalPolicy, self).__init__(need_top_grad=False)
        self.rng = rng
        self.deterministic = deterministic

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
            if self.deterministic:
                out_data[0][:] = mean
            else:
                out_data[0][:] = numpy.sqrt(var.reshape((var.shape[0], 1))) \
                                 * self.rng.randn(*mean.shape) + mean
        else:
            if self.deterministic:
                out_data[0][:] = mean
            else:
                out_data[0][:] = numpy.sqrt(var) * self.rng.randn(*mean.shape) + mean

    def backward(self, out_grad, in_data, out_data, in_grad):
        mean = in_data[0]
        var = in_data[1]
        action = out_data[0]
        score = in_data[2]
        if 1 == var.ndim:
            grad_mu = in_grad[0]
            grad_mu[:] = - (action - mean) * score.reshape((score.shape[0], 1)) / \
                         var.reshape((var.shape[0], 1))
        else:
            grad_mu = in_grad[0]
            grad_var = in_grad[1]
            grad_mu[:] = - (action - mean) * score.reshape((score.shape[0], 1)) / var
            grad_var[:] = - numpy.square(action - mean) * score.reshape((score.shape[0], 1)) \
                          / numpy.square(var) / 2


'''
Name: LogSoftmaxPolicy
Usage: This OP outputs actions generated by a multinomial distribution whose parameters are computed
       by applying softmax to the input.
       The loss function for backward operation is set to be
          \sum_i - log(multinomial(a_i| softmax(x_i)) * R_i
'''


class LogSoftmaxPolicy(mx.operator.NumpyOp):
    def __init__(self, rng=get_numpy_rng(), deterministic=False, implicit_backward=True):
        super(LogSoftmaxPolicy, self).__init__(need_top_grad=False)
        self.rng = rng
        self.deterministic = deterministic
        self.implicit_backward = implicit_backward

    def list_arguments(self):
        if self.implicit_backward:
            return ['data', 'score']
        else:
            return ['data', 'score', 'backward_action']

    def list_outputs(self):
        return ['action', 'prob']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        score_shape = (in_shape[0][0],)
        action_shape = (in_shape[0][0],)
        prob_shape = in_shape[0]
        if self.implicit_backward:
            return [data_shape, score_shape], [action_shape, prob_shape]
        else:
            backward_action_shape = (in_shape[0][0],)
            return [data_shape, score_shape, backward_action_shape], [action_shape, prob_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y_action = out_data[0]
        y_prob = out_data[1]
        y_prob[:] = numpy.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y_prob /= y_prob.sum(axis=1).reshape((x.shape[0], 1))
        if self.deterministic:
            y_action[:] = numpy.argmax(y_prob, axis=1)
        else:
            for ind in range(y_action.shape[0]):
                y_action[ind] = numpy.searchsorted(numpy.cumsum(y_prob[ind]), self.rng.rand())

    def backward(self, out_grad, in_data, out_data, in_grad):
        score = in_data[1]
        if self.implicit_backward:
            action = out_data[0].astype(numpy.int32)
        else:
            action = in_data[2].astype(numpy.int32)
        prob = out_data[1]
        dx = in_grad[0]
        dx[:] = prob
        dx[numpy.arange(action.shape[0]), action] -= 1.0
        dx[:] *= score.reshape(score.shape[0], 1)


'''
Name: LogSoftmaxMaskPolicy
Usage: This OP outputs actions generated by a multinomial distribution whose parameters are computed
       by applying softmax to the input (with mask).
       The loss function for backward operation is set to be
          \sum_i - log(multinomial(a_i| softmax(x_i)) * R_i
'''


class LogSoftmaxMaskPolicy(mx.operator.NumpyOp):
    def __init__(self, rng=get_numpy_rng(), deterministic=False):
        super(LogSoftmaxMaskPolicy, self).__init__(need_top_grad=False)
        self.rng = rng
        self.deterministic = deterministic

    def list_arguments(self):
        return ['data', 'mask', 'score']

    def list_outputs(self):
        return ['action', 'prob']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        mask_shape = in_shape[1]
        assert mask_shape == data_shape, 'Mask Shape and Data Shape must be equal! ' \
                                         'Currently, Mask Shape = %s, Data Shape = %s' \
                                         %(str(mask_shape), str(data_shape))
        score_shape = (in_shape[0][0],)
        action_shape = (in_shape[0][0],)
        prob_shape = in_shape[0]
        return [data_shape, mask_shape, score_shape], \
               [action_shape, prob_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        mask = in_data[1]
        mask = (mask > 0).astype(numpy.float32)
        y_action = out_data[0]
        y_prob = out_data[1]
        y_prob[:] = mask * numpy.exp(x - numpy.ma.array(x, mask=1 - mask).max(axis=1).data.reshape((x.shape[0], 1)))
        y_prob /= y_prob.sum(axis=1).reshape((x.shape[0], 1))
        if self.deterministic:
            y_action[:] = numpy.argmax(y_prob, axis=1)
        else:
            for ind in range(y_action.shape[0]):
                y_action[ind] = numpy.searchsorted(numpy.cumsum(y_prob[ind]), self.rng.rand())

    def backward(self, out_grad, in_data, out_data, in_grad):
        score = in_data[2]
        action = out_data[0].astype(numpy.int32)
        prob = out_data[1]
        dx = in_grad[0]
        dx[:] = prob
        dx[numpy.arange(action.shape[0]), action] -= 1.0
        dx[:] *= score.reshape(score.shape[0], 1)


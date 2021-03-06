from __future__ import absolute_import, division, print_function

import mxnet as mx
import mxnet.ndarray as nd
import numpy
import cv2
from scipy.stats import entropy

from .utils import *


class DQNOutput(mx.operator.CustomOp):
    def __init__(self):
        super(DQNOutput, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # TODO Backward using NDArray will cause some troubles see `https://github.com/dmlc/mxnet/issues/1720'
        x = out_data[0].asnumpy()
        action = in_data[1].asnumpy().astype(numpy.int)
        reward = in_data[2].asnumpy()
        dx = in_grad[0]
        ret = numpy.zeros(shape=dx.shape, dtype=numpy.float32)
        ret[numpy.arange(action.shape[0]), action] \
            = numpy.clip(x[numpy.arange(action.shape[0]), action] - reward, -1, 1)
        self.assign(dx, req[0], ret)


@mx.operator.register("DQNOutput")
class DQNOutputProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DQNOutputProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        reward_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, reward_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return DQNOutput()


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
Name: dqn_sym_nips
Usage: Structure of the Deep Q Network in the NIPS 2013 workshop paper:
      "Playing Atari with Deep Reinforcement Learning" (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
'''


def dqn_sym_nips(action_num, data=None, name='dqn'):
    if data is None:
        net = mx.symbol.Variable('data')
    else:
        net = data
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=action_num)
    net = mx.symbol.Custom(data=net, name=name, op_type='DQNOutput')
    return net


'''
Name: dqn_sym_nature
Usage: Structure of the Deep Q Network in the Nature 2015 paper:
Human-level control through deep reinforcement learning
(http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
'''


def dqn_sym_nature(action_num, data=None, name='dqn'):
    if data is None:
        net = mx.symbol.Variable('data')
    else:
        net = data
    net = mx.symbol.Variable('data')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=64)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv3', kernel=(3, 3), stride=(1, 1), num_filter=64)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=512)
    net = mx.symbol.Activation(data=net, name='relu4', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc5', num_hidden=action_num)
    net = mx.symbol.Custom(data=net, name=name, op_type='DQNOutput')
    return net


class LogNormalPolicy(mx.operator.CustomOp):
    """Outputs actions generated by a policy with normal distribution.
    The loss function for backward operation is set to be
     \sum_i - log(N(a_i|m_i, v_i)) * R_i - H(N(m_i, v_i))
    Here, H(.) is the entropy of the normal distribution, which is
     k/2*(1 + ln(2\pi)) + 1/2 * ln |\Sigma|
    Refer to https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """
    def __init__(self, deterministic, implicit_backward, entropy_regularization, grad_scale):
        super(LogNormalPolicy, self).__init__()
        self.deterministic = deterministic
        self.implicit_backward = implicit_backward
        self.entropy_regularization = entropy_regularization
        self.grad_scale = grad_scale

    def forward(self, is_train, req, in_data, out_data, aux):
        # TODO(sxjscience) There seems to be some problems when I try to use `mx.random.normal`
        # mean = in_data[0]
        # var = in_data[1]
        # if self.deterministic == True:
        #     self.assign(out_data[0], req[0], mean)
        # else:
        #     self.assign(out_data[0], req[0], mean + nd.sqrt(var) * mx.random.normal(0, 1, shape=mean.shape,
        #                                                                             ctx=mean.context))
        mean = in_data[0].asnumpy()
        var = in_data[1].asnumpy()
        if self.deterministic:
            self.assign(out_data[0], req[0], nd.array(mean))
        else:
            self.assign(out_data[0], req[0],
                        nd.array(sample_normal(mean=mean, var=var, rng=get_numpy_rng())))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        mean = in_data[0]
        var = in_data[1]
        if self.implicit_backward:
            action = out_data[0]
        else:
            action = in_data[3]
        score = in_data[2]
        grad_mu = in_grad[0]
        grad_var = in_grad[1]
        self.assign(grad_mu, req[0],
                    - (action - mean) * score.reshape((score.shape[0], 1)) * self.grad_scale / var)
        self.assign(grad_var, req[1], self.grad_scale *
                    ((- nd.square(action - mean) / (2.0 * nd.square(var)) + 1.0 / (2.0 * var)) *
                     score.reshape((score.shape[0], 1)) -
                     numpy.float32(self.entropy_regularization) / (2.0 * var)))


@mx.operator.register("LogNormalPolicy")
class LogNormalPolicyProp(mx.operator.CustomOpProp):
    def __init__(self, deterministic=0, implicit_backward=1, entropy_regularization=0.0, grad_scale=1.0):
        super(LogNormalPolicyProp, self).__init__(need_top_grad=False)
        self.deterministic = safe_eval(deterministic)
        self.implicit_backward = safe_eval(implicit_backward)
        self.entropy_regularization = safe_eval(entropy_regularization)
        self.grad_scale = safe_eval(grad_scale)

    def list_arguments(self):
        if self.implicit_backward:
            return ['mean', 'var', 'score']
        else:
            return ['mean', 'var', 'score', 'backward_action']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        mean_shape = in_shape[0]
        var_shape = in_shape[1]
        score_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        if self.implicit_backward:
            return [mean_shape, var_shape, score_shape], [output_shape], []
        else:
            backward_action_shape = in_shape[0]
            return [mean_shape, var_shape, score_shape, backward_action_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return LogNormalPolicy(deterministic=self.deterministic,
                               implicit_backward=self.implicit_backward,
                               entropy_regularization=self.entropy_regularization,
                               grad_scale=self.grad_scale)


class LogLaplacePolicy(mx.operator.NumpyOp):
    """Outputs actions generated by a policy with laplace distribution.
    The loss function for backward operation is set to be
     \sum_i - log(Laplace(a_i|\mu_i, b_i)) * R_i
    Laplace Distribution: \frac{1}{2b} exp(-\frac{\abs{x - \mu}{b})
    """
    def __init__(self, rng=get_numpy_rng(), deterministic=False):
        super(LogLaplacePolicy, self).__init__(need_top_grad=False)
        self.rng = rng
        self.deterministic = deterministic

    def list_arguments(self):
        return ['mean', 'scale', 'score']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        mean_shape = in_shape[0]
        scale_shape = in_shape[1]
        score_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [mean_shape, scale_shape, score_shape], \
               [output_shape]

    def forward(self, in_data, out_data):
        mean = in_data[0]
        scale = in_data[1]
        if self.deterministic:
            out_data[0][:] = mean
        else:
            out_data[0][:] = self.rng.laplace(loc=mean, scale=scale)

    def backward(self, out_grad, in_data, out_data, in_grad):
        mean = in_data[0]
        scale = in_data[1]
        action = out_data[0]
        score = in_data[2]
        grad_mu = in_grad[0]
        grad_scale = in_grad[1]
        grad_mu[:] = numpy.sign(mean - action) / scale * score.reshape((score.shape[0], 1))
        grad_scale[:] = (- numpy.abs(mean - action) / (scale * scale) + 1.0 / scale) * \
                        score.reshape((score.shape[0], 1))


class LogSoftmaxPolicy(mx.operator.CustomOp):
    """Outputs actions generated by a multinomial distribution whose parameters are computed
    by applying softmax to the input.
    Loss function for backward operation is set to be
      \sum_i - log(multinomial(a_i| softmax(x_i)) * R_i - H(multinomial(a_i|softmax(x_i)))
    """
    def __init__(self, deterministic, implicit_backward, use_mask, entropy_regularization, grad_scale):
        super(LogSoftmaxPolicy, self).__init__()
        self.deterministic = deterministic
        self.implicit_backward = implicit_backward
        self.use_mask = use_mask
        self.entropy_regularization = entropy_regularization
        self.grad_scale = grad_scale

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y_prob = out_data[1].asnumpy()
        if self.use_mask:
            mask = in_data[1].asnumpy()
            mask = (mask > 0).astype(numpy.float32)
            y_prob[:] = mask * numpy.exp(x - numpy.ma.array(x, mask=1 - mask).max(axis=1).data.reshape((x.shape[0], 1)))
        else:
            y_prob[:] = numpy.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y_prob /= y_prob.sum(axis=1).reshape((x.shape[0], 1))
        if self.deterministic:
            self.assign(out_data[0], req[0], nd.array(numpy.argmax(y_prob, axis=1)))
            self.assign(out_data[1], req[1], nd.array(y_prob))
        else:
            y_action = sample_categorical(prob=y_prob, rng=get_numpy_rng())
            self.assign(out_data[0], req[0], nd.array(y_action))
            self.assign(out_data[1], req[1], nd.array(y_prob))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if self.use_mask:
            score = in_data[2].asnumpy()
        else:
            score = in_data[1].asnumpy()
        if self.implicit_backward:
            action = out_data[0].asnumpy().astype(numpy.int32)
        else:
            action = in_data[2].asnumpy().astype(numpy.int32)
        prob = out_data[1].asnumpy()
        dx = in_grad[0].asnumpy()
        dx[:] = prob
        dx[numpy.arange(action.shape[0]), action] -= 1.0
        dx[:] *= score.reshape(score.shape[0], 1)
        dx[:] += self.entropy_regularization * (prob * (entropy(prob.T).reshape(prob.shape[0], 1)) +
                                                numpy.nan_to_num(prob * numpy.log(prob)))
        self.assign(in_grad[0], req[0], self.grad_scale * dx)


@mx.operator.register("LogSoftmaxPolicy")
class LogSoftmaxPolicyProp(mx.operator.CustomOpProp):
    def __init__(self, deterministic=False, implicit_backward=True, use_mask=False,
                 entropy_regularization=0.01, grad_scale=1.0):
        super(LogSoftmaxPolicyProp, self).__init__(need_top_grad=False)
        self.deterministic = safe_eval(deterministic)
        self.implicit_backward = safe_eval(implicit_backward)
        self.use_mask = safe_eval(use_mask)
        self.entropy_regularization = safe_eval(entropy_regularization)
        self.grad_scale = safe_eval(grad_scale)

    def list_arguments(self):
        if self.implicit_backward:
            if self.use_mask:
                return ['data', 'mask', 'score']
            else:
                return ['data', 'score']
        else:
            if self.use_mask:
                return ['data', 'mask', 'score', 'backward_action']
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
            if self.use_mask:
                mask_shape = in_shape[0]
                return [data_shape, mask_shape, score_shape], [action_shape, prob_shape], []
            else:
                return [data_shape, score_shape], [action_shape, prob_shape], []
        else:
            backward_action_shape = (in_shape[0][0],)
            if self.use_mask:
                mask_shape = in_shape[0]
                return [data_shape, mask_shape, score_shape, backward_action_shape], [action_shape, prob_shape], []
            else:
                return [data_shape, score_shape, backward_action_shape], [action_shape, prob_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return LogSoftmaxPolicy(deterministic=self.deterministic,
                                implicit_backward=self.implicit_backward,
                                use_mask=self.use_mask,
                                entropy_regularization=self.entropy_regularization,
                                grad_scale=self.grad_scale)


class LogMoGPolicy(mx.operator.CustomOp):
    """Outputs actions generated by a mixture of gaussian distribution.
    The loss function for backward operation is set to be
        \sum_j (- log(\sum_i \alpha_i N(x_j|\mu_i, \lambda_i)) * R_j)
    More Details:
        prob --> (batch_num, center_num)
        mean --> (batch_num, center_num, sample_dim)
        var  --> (batch_num, center_num, sample_dim)

        output --> (batch_num, sample_dim)
    If the `deterministic` flag is set, output is set to be the mode of the distribution.
    """
    def __init__(self, deterministic, implicit_backward, grad_scale):
        super(LogMoGPolicy, self).__init__()
        self.deterministic = deterministic
        self.implicit_backward = implicit_backward
        self.grad_scale = grad_scale

    def forward(self, is_train, req, in_data, out_data, aux):
        prob = in_data[0].asnumpy()
        mean = in_data[1].asnumpy()
        var = in_data[2].asnumpy()
        if self.deterministic:
            self.assign(out_data[0], req[0], nd.array(mean[numpy.arange(mean.shape[0]),
                                                      numpy.argmax(prob, axis=1), :]))
        else:
            self.assign(out_data[0], req[0],
                        nd.array(sample_mog(prob=prob, mean=mean, var=var, rng=get_numpy_rng()),
                                 out_data[0].context))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        prob = in_data[0].asnumpy()
        mean = in_data[1].asnumpy()
        var = in_data[2].asnumpy()
        if self.implicit_backward:
            action = out_data[0].asnumpy().reshape((mean.shape[0], 1, mean.shape[2]))
        else:
            action = in_data[4].asnumpy().reshape((mean.shape[0], 1, mean.shape[2]))
        score = in_data[3].asnumpy()
        lognormals = numpy.sum(- numpy.square(mean - action)/(2.0 * var) - numpy.log(var)/2.0, axis=2)

        # Deal with potential underflow problem similar to softmax
        normals = numpy.exp(lognormals - lognormals.max(axis=1).reshape((lognormals.shape[0], 1)))
        weighted_normals = (normals * prob).sum(axis=1).reshape(mean.shape[0], 1)
        grad_prob = - normals / weighted_normals * score.reshape(mean.shape[0], 1) * self.grad_scale
        grad_mean = - (normals * prob / weighted_normals).reshape(prob.shape + (1,)) * \
                     (action - mean) / var * score.reshape(mean.shape[0], 1, 1) * self.grad_scale
        grad_var = - (normals * prob / weighted_normals).reshape(prob.shape + (1,)) * \
                    (numpy.square(mean - action)/(2.0 * numpy.square(var)) - 0.5/var) * \
                   score.reshape(mean.shape[0], 1, 1) * self.grad_scale
        self.assign(in_grad[0], req[0], nd.array(grad_prob))
        self.assign(in_grad[1], req[1], nd.array(grad_mean))
        self.assign(in_grad[2], req[2], nd.array(grad_var))


@mx.operator.register("LogMoGPolicy")
class LogMoGPolicyProp(mx.operator.CustomOpProp):
    def __init__(self, deterministic=0, implicit_backward=1, grad_scale=1.0):
        super(LogMoGPolicyProp, self).__init__(need_top_grad=False)
        self.deterministic = safe_eval(deterministic)
        self.implicit_backward = safe_eval(implicit_backward)
        self.grad_scale = safe_eval(grad_scale)

    def list_arguments(self):
        if self.implicit_backward:
            return ['prob', 'mean', 'var', 'score']
        else:
            return ['prob', 'mean', 'var', 'score', 'backward_action']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        prob_shape = in_shape[0]
        mean_shape = in_shape[1]
        assert 3 == len(mean_shape)
        var_shape = in_shape[1]
        score_shape = (in_shape[0][0],)
        output_shape = (in_shape[1][0], in_shape[1][2])
        assert mean_shape[0] == prob_shape[0] and score_shape[0] == mean_shape[0]
        if self.implicit_backward:
            return [prob_shape, mean_shape, var_shape, score_shape], [output_shape], []
        else:
            backward_action_shape = output_shape
            return [prob_shape, mean_shape, var_shape, score_shape, backward_action_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return LogMoGPolicy(deterministic=self.deterministic,
                            implicit_backward=self.implicit_backward,
                            grad_scale=self.grad_scale)


#TODO Add HOG layer: hog = cv2.HOGDescriptor()

class ArenaSym(object):
    @staticmethod
    def spatial_softmax(data, channel_num, rows, cols, name=None):
        out = mx.symbol.Reshape(data, shape=(-1, rows*cols))
        out = mx.symbol.SoftmaxActivation(data=out, mode='instance')
        out = mx.symbol.Reshape(data=out, shape=(-1, channel_num, rows, cols), name=name)
        return out

    @staticmethod
    def normalize_channel(data, axis, name=None):
        out = mx.symbol.sum(mx.symbol.square(data), axis=axis, keepdims=True)
        out = mx.symbol.broadcast_div(data, mx.symbol.sqrt(out) + 1E-8, name=name)
        return out

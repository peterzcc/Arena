from __future__ import absolute_import, division, print_function
import logging
import mxnet as mx
import numpy as np
import scipy.stats
from ..utils import *

class IdentityOp(mx.operator.CustomOp):
    def __init__(self, logging_prefix="identity", input_debug=False, grad_debug=False):
        super(IdentityOp, self).__init__()
        self.logging_prefix=logging_prefix
        self.input_debug = input_debug
        self.grad_debug = grad_debug

    def forward(self, is_train, req, in_data, out_data, aux):
        if(self.input_debug):
            logging.debug("%s: in_norm=%f, in_shape=%s"
                          %(self.logging_prefix, np.linalg.norm(in_data[0].asnumpy()), str(in_data[0].shape)))
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if (self.grad_debug):
            logging.debug("%s: grad_norm=%f, grad_shape=%s"
                          % (self.logging_prefix, np.linalg.norm(out_grad[0].asnumpy()), str(out_grad[0].shape)))
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("identity")
class IdentityOpProp(mx.operator.CustomOpProp):
    def __init__(self, logging_prefix="identity", input_debug=False, grad_debug=False):
        super(IdentityOpProp, self).__init__(need_top_grad=True)
        self.input_debug = safe_eval(input_debug)
        self.grad_debug = safe_eval(grad_debug)
        self.logging_prefix = str(logging_prefix)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return IdentityOp(input_debug=self.input_debug,
                          grad_debug=self.grad_debug,
                          logging_prefix=self.logging_prefix)

class ConstantOp(mx.operator.CustomOp):
    """Implementation of mask on minibatch layer.
    """
    def __init__(self, data):
        super(ConstantOp, self).__init__()
        self.data = data

    def forward(self, is_train, req, in_data, out_data, aux):
        if self.data.context != out_data[0].context:
            self.data = self.data.copyto(out_data[0].context)
        self.assign(out_data[0], req[0], self.data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise RuntimeError("cannot bp to constant")


@mx.operator.register("constant")
class ConstantOpProp(mx.operator.CustomOpProp):
    def __init__(self, pkl_data):
        super(ConstantOpProp, self).__init__(need_top_grad=False)
        self.data = pickle.loads(pkl_data)

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [self.data.shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return ConstantOp(mx.nd.array(self.data))


class LogisticRegressionMaskOutput(mx.operator.CustomOp):
    def __init__(self, ignore_label):
        super(LogisticRegressionMaskOutput, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], 1.0 / (1.0 + nd.exp(- in_data[0])))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        output = out_data[0].asnumpy()
        label = in_data[1].asnumpy()
        data_grad = (output - label) * (label != self.ignore_label)
        self.assign(in_grad[0], req[0], data_grad)

@mx.operator.register("LogisticRegressionMaskOutput")
class LogisticRegressionMaskOutputProp(mx.operator.CustomOpProp):
    def __init__(self, ignore_label):
        super(LogisticRegressionMaskOutputProp, self).__init__(need_top_grad=False)
        self.ignore_label = safe_eval(ignore_label)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return LogisticRegressionMaskOutput(ignore_label=self.ignore_label)

class EntropyMultinomialDist(mx.operator.CustomOp):
    def __init__(self):
        super(EntropyMultinomialDist, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], scipy.stats.entropy(in_data[0].asnumpy().T))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        p = in_data[0]
        p_sum = nd.sum(p, axis=1, keepdims=True)
        logit = nd.log(p / p_sum)
        grad = - logit / p_sum + nd.sum(p * logit, axis=1, keepdims=True) / nd.square(p_sum)
        grad[:] = nd.expand_dims(out_grad[0], axis=1) * grad
        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("entropy_multinomial")
class EntropyMultinomialDistProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(EntropyMultinomialDistProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (in_shape[0][0],)
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return EntropyMultinomialDist()


def logistic_regression_mask_output(data, label, ignore_label, name=None):
    return mx.sym.Custom(name=name,
                         op_type="LogisticRegressionMaskOutput",
                         ignore_label=ignore_label,
                         data=data,
                         label=label)


def constant(data, name="constant"):
    if isinstance(data, mx.nd.NDArray):
        data = data.asnumpy()
    pkl_data = pickle.dumps(data)
    return mx.symbol.Custom(name=name,
                            op_type="constant",
                            pkl_data=pkl_data)


def identity(data, name="identity", logging_prefix=None,
             input_debug=False, grad_debug=False):
    return mx.symbol.Custom(data=data,
                            name=name,
                            logging_prefix=name,
                            input_debug=input_debug,
                            grad_debug=grad_debug,
                            op_type="identity")


def entropy_multinomial(data, name="entropy"):
    return mx.symbol.Custom(name=name,
                            op_type="entropy_multinomial",
                            data=data)

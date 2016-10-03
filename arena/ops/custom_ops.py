import mxnet as mx
import numpy
from ..utils import *

class IdentityOp(mx.operator.CustomOp):
    def __init__(self):
        super(IdentityOp, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        # print(in_data[0].shape)
        # print("in_data", in_data[0].asnumpy())
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # print("out_grad shape:", out_grad[0].shape)
        # print("in_grad shape:", in_grad[0].shape)
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("identity")
class IdentityOpProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(IdentityOpProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return IdentityOp()

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
        print(self.ignore_label)

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


def identity(data, name="identity"):
    return mx.symbol.Custom(data=data,
                            name=name,
                            op_type="identity")
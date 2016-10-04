import mxnet as mx
import numpy

class Identity(mx.operator.CustomOp):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        # print(in_data[0].shape)
        # print("in_data", in_data[0].asnumpy())
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # print("out_grad shape:", out_grad[0].shape)
        # print("in_grad shape:", in_grad[0].shape)
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("Identity")
class IdentityProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(IdentityProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return Identity()
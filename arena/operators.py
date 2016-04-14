import mxnet as mx
import mxnet.ndarray as nd
import numpy
import pyfftw

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

class FFTNpyOp(mx.operator.NumpyOp):
    def __init__(self):
        super(FFTNpyOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['real', 'imag']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        real_shape = in_shape[0]
        imag_shape = in_shape[0]
        return [data_shape], [real_shape, imag_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        real_part = out_data[0]
        imag_part = out_data[1]
        res = pyfftw.interfaces.numpy_fft.fft2(x)
        real_part[:] = res.real
        imag_part[:] = res.imag

class IFFTNpyOp(mx.operator.NumpyOp):
    def __init__(self):
        super(IFFTNpyOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['real', 'imag']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        real_shape = in_shape[0]
        imag_shape = in_shape[1]
        output_shape = in_shape[0]

        return [real_shape, imag_shape], output_shape

    def forward(self, in_data, out_data):
        real_part = in_data[0]
        imag_part = in_data[1]
        ifft_res = out_data[0]
        ifft_res[:] = pyfftw.interfaces.numpy_fft.ifft2(numpy.vectorize(complex)(real_part,
                                                                                 imag_part))

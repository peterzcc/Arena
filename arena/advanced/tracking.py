import numpy
import mxnet as mx

class MeshGridNpyOp(mx.operator.NumpyOp):
    def __init__(self, x_linspace, y_linspace):
        super(MeshGridNpyOp, self).__init__(need_top_grad=False)
        self.x_linspace = x_linspace
        self.y_linspace = y_linspace

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['y_ind_mat', 'x_ind_mat']

    def infer_shape(self, in_shape):
        y_ind_mat_shape = (len(self.y_linspace), len(self.x_linspace))
        x_ind_mat_shape = (len(self.y_linspace), len(self.x_linspace))
        return [], [y_ind_mat_shape, x_ind_mat_shape]

    def forward(self, in_data, out_data):
        y_ind_mat = out_data[0]
        x_ind_mat = out_data[1]
        x_ind_mat[:], y_ind_mat[:] = numpy.meshgrid(self.y_linspace, self.x_linspace)

class GaussianMapGeneratorOp(mx.operator.NumpyOp):
    def __init__(self, sigma_factor, rows, cols):
        super(GaussianMapGeneratorOp, self).__init__(need_top_grad=False)
        self.sigma_factor = sigma_factor
        self.rows = rows
        self.cols = cols
        x_ind_mat, y_ind_mat = numpy.meshgrid(numpy.linspace(-self.cols/2, self.cols/2, self.cols),
                                                        numpy.linspace(-self.rows/2, self.rows/2,
                                                                       self.rows))
        self.distance = numpy.square(x_ind_mat).astype(numpy.float32) + \
                        numpy.square(y_ind_mat).astype(numpy.float32)

    def list_arguments(self):
        return ['attention_size', 'object_size']

    def list_outputs(self):
        return ['gaussian_map']

    def infer_shape(self, in_shape):
        attention_size_shape = in_shape[0]
        object_size_shape = in_shape[1]
        assert object_size_shape == attention_size_shape
        gaussian_map_shape = (in_shape[0][0], 1, self.rows, self.cols)
        return [attention_size_shape, object_size_shape], [gaussian_map_shape]

    def forward(self, in_data, out_data):
        attention_size = in_data[0]
        object_size = in_data[1]
        gaussian_map = out_data[0]
        for i in range(attention_size.shape[0]):
            ratio = numpy.sqrt(numpy.prod(object_size[i]/attention_size[i]).astype(numpy.float32))
            gaussian_map[i, 0, :, :] = numpy.exp(-self.distance / 2 /
                                                 numpy.square(ratio * self.sigma_factor))


def gaussian_map_fft(attention_size, object_size, timestamp, sigma_factor, rows, cols):
    gaussian_map_op = GaussianMapGeneratorOp(sigma_factor=sigma_factor, rows=rows, cols=cols)
    ret = gaussian_map_op(attention_size=attention_size, object_size=object_size)
    ret = mx.symbol.FFT2D(data=ret, name="GaussianMapFFT_t%d" % timestamp)
    ret = mx.symbol.BlockGrad(data=ret)
    return ret


class CorrelationFilterHandler(object):
    def __init__(self, rows, cols, gaussian_sigma_factor, regularizer):
        super(CorrelationFilterHandler).__init__()
        self.rows = rows
        self.cols = cols
        self.sigma_factor = gaussian_sigma_factor
        self.regularizer = regularizer

    def get_embedding(self, attention_element, feature_extractor):
        return

    def get_joint_embedding(self, memory_element, attention_element, feature_extractor):
        return

def perceive():
    return None


def get_correlation_filter_template(image, roi, perceive_function):
    feature_maps = perceive(image, roi)
    return None

def compute_correlation_filter_template():
    return None
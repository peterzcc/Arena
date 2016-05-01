import numpy
import mxnet as mx
from arena.helpers.pretrained import vgg_m
from .common import *
from collections import namedtuple, OrderedDict
from .attention import get_multiscale_size


'''

ScaleCFTemplate --> The basic memory structure
numerators: (num_memory, scale_num * channel of single template, row, col)
denominators: (num_memory, scale_num * channel of single template, row, col)
state: list of LSTMState (Multiple layers)
status: Status like reading times and visiting timestamp of each memory cell

MemoryStat --> The statistical variables of the memory
counter: Counter of the memory
visiting_timestamp: The recorded visiting timestamp of the memory elements
'''

ScaleCFTemplate = namedtuple("ScaleCFTemplate", ["numerator", "denominator"])


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
    def __init__(self, sigma_factors, rows, cols):
        super(GaussianMapGeneratorOp, self).__init__(need_top_grad=False)
        self.sigma_factors = sigma_factors
        self.rows = rows
        self.cols = cols
        x_ind_mat, y_ind_mat = numpy.meshgrid(
            numpy.linspace(-self.cols / 2, self.cols / 2, self.cols),
            numpy.linspace(-self.rows / 2, self.rows / 2,
                           self.rows))
        self.distance = numpy.square(x_ind_mat).astype(numpy.float32) + \
                        numpy.square(y_ind_mat).astype(numpy.float32)

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['gaussian_map']

    def infer_shape(self, in_shape):
        gaussian_map_shape = (len(self.sigma_factors), 1, self.rows, self.cols)
        return [], [gaussian_map_shape]

    def forward(self, in_data, out_data):
        gaussian_map = out_data[0]
        for i, sigma_factor in enumerate(self.sigma_factors):
            gaussian_map[i, 0, :, :] = numpy.exp(-self.distance / (2 * numpy.square(sigma_factor)))


class HannWindowGeneratorOp(mx.operator.NumpyOp):
    def __init__(self, rows, cols):
        super(HannWindowGeneratorOp, self).__init__(need_top_grad=False)
        self.rows = rows
        self.cols = cols
        self.hann_window = numpy.dot(numpy.hanning(rows).reshape((rows, 1)),
                                     numpy.hanning(cols).reshape((1, cols)))
    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['hanning_window']

    def infer_shape(self, in_shape):
        return [], [(1,1) + self.hann_window.shape]

    def forward(self, in_data, out_data):
        hanning_map = out_data[0]
        hanning_map[:] = self.hann_window


class CorrelationFilterHandler(object):
    def __init__(self, rows, cols, gaussian_sigma_factor, regularizer, perception_handler,
                 glimpse_handler):
        super(CorrelationFilterHandler, self).__init__()
        self.rows = numpy.int32(rows)
        self.cols = numpy.int32(cols)
        self.out_rows = self.rows
        self.out_cols = (self.cols /2 + 1) * 2
        self.gaussian_sigma_factor = gaussian_sigma_factor
        self.regularizer = regularizer
        self.perception_handler = perception_handler
        self.glimpse_handler = glimpse_handler
        hannmap_op = HannWindowGeneratorOp(rows=self.rows, cols=self.cols)
        self.hannmap = hannmap_op()
        self.hannmap = mx.symbol.BroadcastChannel(self.hannmap, dim=0, size=self.scale_num)
        self.hannmap = mx.symbol.BroadcastChannel(self.hannmap, dim=1, size=self.channel_size)

        # Build the sigma_factor list
        self.sigma_factors = []
        sigma_factor = self.gaussian_sigma_factor / glimpse_handler.init_scale
        for i in range(self.scale_num):
            self.sigma_factors.append(sigma_factor)
            sigma_factor /= glimpse_handler.scale_mult
        gaussian_map_op = GaussianMapGeneratorOp(sigma_factors=self.sigma_factors,
                                                 rows=rows, cols=cols)
        self.gaussian_map_fft = mx.symbol.FFT2D(data=gaussian_map_op())
        self.gaussian_map_fft = mx.symbol.BroadcastChannel(self.gaussian_map_fft,
                                                           dim=1, size=self.channel_size)

    @property
    def name(self):
        return "CorrelationFilter"

    @property
    def scale_num(self):
        return self.glimpse_handler.scale_num

    @property
    def channel_size(self):
        return self.perception_handler.channel_size

    def get_multiscale_template(self, glimpse, postfix=''):
        multiscale_feature = self.perception_handler.perceive(
            data_sym=glimpse.data,
            name=self.name + ":multiscale_feature" + postfix) * self.hannmap
        multiscale_feature_fft = mx.symbol.FFT2D(multiscale_feature)

        numerator = mx.symbol.ComplexHadamard(self.gaussian_map_fft,
                                              mx.symbol.Conjugate(multiscale_feature_fft))
        denominator = mx.symbol.ComplexHadamard(mx.symbol.Conjugate(multiscale_feature_fft),
                                                multiscale_feature_fft) + \
                      mx.symbol.ComplexHadamard(multiscale_feature_fft,
                                                mx.symbol.ComplexExchange(multiscale_feature_fft))
        denominator = mx.symbol.SumChannel(denominator)
        numerator = mx.symbol.BlockGrad(numerator, name=(self.name + ':numerator' + postfix))
        denominator = mx.symbol.BlockGrad(denominator, name=(self.name + ':denominator' + postfix))
        multiscale_template = ScaleCFTemplate(numerator=numerator, denominator=denominator)
        return multiscale_template

    def get_multiscale_scoremap(self, multiscale_template, glimpse, postfix=''):
        multiscale_feature = self.perception_handler.perceive(
            data_sym=glimpse.data,
            name=self.name + ":multiscale_feature" + postfix) * self.hannmap

        multiscale_feature_fft = mx.symbol.FFT2D(multiscale_feature)
        numerator = multiscale_template.numerator
        denominator = mx.symbol.BroadcastChannel(data=multiscale_template.denominator, dim=1,
                                                 size=self.channel_size)
        processed_template = numerator / (denominator + self.regularizer)
        multiscale_scoremap = mx.symbol.IFFT2D(data=mx.symbol.ComplexHadamard(processed_template,
                                                                              multiscale_feature_fft),
                                               output_shape=(self.rows, self.cols))
        multiscale_scoremap = mx.symbol.BlockGrad(multiscale_scoremap,
                                                  name=self.name + ':multiscale_scoremap' + postfix)
        return multiscale_scoremap

class PerceptionHandler(object):
    def __init__(self, net_type='VGG-M', blocked=True):
        super(PerceptionHandler, self).__init__()
        self.net_type = 'VGG-M'
        self.blocked = blocked
        self.params_data, self.params_sym = self.load_net(self.net_type)

    def load_net(self, net_type):
        if 'VGG-M' == net_type:
            from arena.helpers.pretrained import vgg_m
            params = vgg_m()
            params_sym = {k: mx.symbol.Variable(k) for k in params}
        else:
            raise NotImplementedError
        return params, params_sym

    def set_params(self, params):
        for k, v in params.items():
            if k == 'arg:conv1_weight' or k == 'arg:conv1_bias':
                v[:] = self.params_data[k]

    @property
    def channel_size(self):
        if 'VGG-M' == self.net_type:
            return 96

    def perceive(self, data_sym, name=''):
        if 'VGG-M' == self.net_type:
            if not self.blocked:
                conv1 = mx.symbol.Convolution(data=data_sym,
                                              weight=self.params_sym['arg:conv1_weight'],
                                              bias=self.params_sym['arg:conv1_bias'], kernel=(7, 7),
                                              stride=(2, 2), num_filter=96)
            else:
                conv1 = mx.symbol.Convolution(data=data_sym, weight=self.params_sym['arg:conv1_weight'],
                                              bias=self.params_sym['arg:conv1_bias'], kernel=(7, 7),
                                              stride=(2, 2), num_filter=96)
                conv1 = mx.symbol.BlockGrad(data=conv1, name=name)
            return conv1
        else:
            raise NotImplementedError


class ScoreMapProcessor(object):
    def __init__(self, dim_in, num_filter=4, scale_num=1):
        super(ScoreMapProcessor, self).__init__()
        self.num_filter = num_filter
        self.dim_in = dim_in
        self.scale_num = scale_num
        self.params = self._init_params()

    def _init_params(self):
        params = OrderedDict()
        for i in range(self.scale_num):
            params[self.name + ':scale%d:conv1' %i] = ConvParam(
                weight=mx.symbol.Variable(self.name + ':scale%d:conv1_weight' %i),
                bias=mx.symbol.Variable(self.name + ':scale%d:conv1_bias' %i))
            params[self.name + ':scale%d:conv2' %i] = ConvParam(
                weight=mx.symbol.Variable(self.name + ':scale%d:conv2_weight' %i),
                bias=mx.symbol.Variable(self.name + ':scale%d:conv2_bias' %i))
        return params

    @property
    def dim_out(self):
        return (self.num_filter, self.dim_in[1]/2, self.dim_in[2]/2)

    @property
    def name(self):
        return "ScoreMapProcessor"

    def scoremap_processing(self, multiscale_scoremap, postfix=''):
        multiscale_scoremap = mx.symbol.SliceChannel(multiscale_scoremap,
                                                     num_outputs=self.scale_num,
                                                     axis=0)
        parsed_scoremaps = []
        for i in range(self.scale_num):
            conv1 = mx.symbol.Convolution(data=multiscale_scoremap[i],
                                          weight=self.params[self.name + ':scale%d:conv1' %i].weight,
                                          bias=self.params[self.name + ':scale%d:conv1' %i].bias,
                                          kernel=(3,3), pad=(1,1),
                                          num_filter=self.num_filter,
                                          name=self.name + (':scale%d:conv1' %i) + postfix)
            act1 = mx.symbol.Activation(data=conv1, act_type='relu', name=self.name + (':scale%d:act1' %i) + postfix) + conv1
            pool1 = mx.symbol.Pooling(data=act1, kernel=(2, 2), pool_type='avg', stride=(2, 2))
            conv2 = mx.symbol.Convolution(data=pool1,
                                          weight=self.params[self.name + ':scale%d:conv2' %i].weight,
                                          bias=self.params[self.name + ':scale%d:conv2' %i].bias,
                                          kernel=(3, 3), pad=(1, 1),
                                          num_filter=self.num_filter,
                                          name=self.name + (':scale%d:conv2' %i) + postfix)
            act2 = mx.symbol.Activation(data=conv2, act_type='relu', name=self.name +
                                                                          (':scale%d:act2' %i) +
                                                                          postfix) + conv2
            parsed_scoremaps.append(act2)
        return mx.symbol.Concat(*parsed_scoremaps, num_args=self.scale_num, dim=1)

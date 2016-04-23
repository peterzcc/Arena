import numpy
import mxnet as mx
from arena.helpers.pretrained import vgg_m
from collections import namedtuple

CFTemplate = namedtuple("CFTemplate", ["numerator", "denominator"])


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
        x_ind_mat, y_ind_mat = numpy.meshgrid(
            numpy.linspace(-self.cols / 2, self.cols / 2, self.cols),
            numpy.linspace(-self.rows / 2, self.rows / 2,
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
            ratio = numpy.sqrt(numpy.prod(object_size[i] / attention_size[i]).astype(numpy.float32))
            gaussian_map[i, 0, :, :] = numpy.exp(-self.distance / 2 /
                                                 numpy.square(ratio * self.sigma_factor))

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


def gaussian_map_fft(attention_size, object_size, sigma_factor, rows, cols, prefix="", postfix=""):
    gaussian_map_op = GaussianMapGeneratorOp(sigma_factor=sigma_factor, rows=rows, cols=cols)
    ret = gaussian_map_op(attention_size=attention_size, object_size=object_size)
    ret = mx.symbol.FFT2D(data=ret)
    ret = mx.symbol.BlockGrad(data=ret, name="%sGaussianMapFFT%s" % (prefix, postfix))
    return ret


class CorrelationFilterHandler(object):
    def __init__(self, rows, cols, gaussian_sigma_factor, regularizer, perception_handler,
                 batch_size=1):
        super(CorrelationFilterHandler, self).__init__()
        self.rows = numpy.int32(rows)
        self.cols = numpy.int32(cols)
        self.sigma_factor = gaussian_sigma_factor
        self.regularizer = regularizer
        self.batch_size = batch_size
        self.perception_handler = perception_handler
        hannmap_op = HannWindowGeneratorOp(rows=self.rows, cols=self.cols)
        self.hannmap = hannmap_op()
        self.hannmap = mx.symbol.BroadcastChannel(self.hannmap, dim=0, size=self.batch_size)
        self.hannmap = mx.symbol.BroadcastChannel(self.hannmap, dim=1, size=self.channel_size)

    @property
    def name(self):
        return "CorrelationFilter"

    @property
    def channel_size(self):
        return self.perception_handler.channel_size

    def get_multiscale_template(self, glimpse, object_size, timestamp=0, attention_step=0):
        multiscale_template = []
        features = []
        for i, image_patch in enumerate(glimpse):
            postfix = "scale%d_t%d_step%d" % (i, timestamp, attention_step)
            features.append(self.perception_handler.perceive(image_patch.data, postfix) *
                            self.hannmap)
        feature_ffts = mx.symbol.FFT2D(mx.symbol.Concat(*features, dim=0))
        feature_ffts = mx.symbol.SliceChannel(feature_ffts, num_outputs=len(glimpse), axis=0)

        #TODO We can accelerate this part by merging them into a single batch
        for i, image_patch in enumerate(glimpse):
            gaussian_map = gaussian_map_fft(attention_size=image_patch.size,
                                            object_size=object_size,
                                            sigma_factor=self.sigma_factor,
                                            rows=self.rows, cols=self.cols,
                                            postfix=postfix)
            gaussian_map = mx.symbol.BroadcastChannel(gaussian_map, dim=1, size=self.channel_size)
            numerator = mx.symbol.ComplexHadamard(gaussian_map,
                                                  mx.symbol.Conjugate(feature_ffts[i]))
            denominator = mx.symbol.ComplexHadamard(mx.symbol.Conjugate(feature_ffts[i]),
                                                    feature_ffts[i]) + \
                          mx.symbol.ComplexHadamard(feature_ffts[i],
                                                    mx.symbol.ComplexExchange(feature_ffts[i]))
            denominator = mx.symbol.SumChannel(denominator) + self.regularizer
            numerator = mx.symbol.BlockGrad(numerator, name='numerator' + postfix)
            denominator = mx.symbol.BlockGrad(denominator, name='denominator' + postfix)
            multiscale_template.append(CFTemplate(numerator, denominator))
        return multiscale_template

    def get_joint_embedding(self, multiscale_template, glimpse, timestamp=0, attention_step=0):
        assert len(multiscale_template) == len(glimpse)
        scores = []
        features = []
        for i, image_patch in enumerate(glimpse):
            postfix = "scale%d_t%d_step%d" % (i, timestamp, attention_step)
            features.append(self.perception_handler.perceive(image_patch.data, postfix) *
                            self.hannmap)
        feature_ffts = mx.symbol.FFT2D(mx.symbol.Concat(*features, dim=0))

        numerators = []
        denominators = []
        for i, template in enumerate(multiscale_template):
            numerators.append(template.numerator)
            denominators.append(mx.symbol.BroadcastChannel(data=template.denominator, dim=1,
                                                           size=self.channel_size))
        processed_template = mx.symbol.Concat(*numerators, dim=0)/\
                             mx.symbol.Concat(*denominators, dim=0)
        scores = mx.symbol.IFFT2D(data=mx.symbol.ComplexHadamard(processed_template, feature_ffts),
                                  output_shape=(self.rows, self.cols))
        scores = mx.symbol.SliceChannel(scores, num_outputs=len(glimpse), axis=0)
        score_maps = []
        for i in range(len(glimpse)):
            postfix = "scale%d_t%d_step%d" % (i, timestamp, attention_step)
            score_map = mx.symbol.BlockGrad(scores[i], name='scoremap' + postfix)
            score_maps.append(scores[i])
        return score_maps

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

    def perceive(self, data_sym, postfix=''):
        if 'VGG-M' == self.net_type:
            if not self.blocked:
                conv1 = mx.symbol.Convolution(data=data_sym,
                                              weight=self.params_sym['arg:conv1_weight'],
                                              bias=self.params_sym['arg:conv1_bias'], kernel=(7, 7),
                                              stride=(2, 2), num_filter=96,
                                              name="%s-Perceive%s" %(self.net_type, postfix))
            else:
                conv1 = mx.symbol.Convolution(data=data_sym, weight=self.params_sym['arg:conv1_weight'],
                                              bias=self.params_sym['arg:conv1_bias'], kernel=(7, 7),
                                              stride=(2, 2), num_filter=96,
                                              name="%s-Perceive%s" % (self.net_type, postfix))
                conv1 = mx.symbol.BlockGrad(data=conv1, name="perception%s")
            return conv1
        else:
            raise NotImplementedError

class ScoreMapProcessor(object):
    def __init__(self, dim_in, num_filter=64):
        super(ScoreMapProcessor, self).__init__()
        self.num_filter = num_filter
        self.dim_in = dim_in.copy()
        self.params = self._init_params()

    def _init_params(self):
        params = {}
        params[self.name + ':conv1_weight'] = mx.symbol.Variable(self.name + ':conv1_weight')
        params[self.name + ':conv1_bias'] = mx.symbol.Variable(self.name + ':conv1_bias')
        params[self.name + ':conv2_weight'] = mx.symbol.Variable(self.name + ':conv2_weight')
        params[self.name + ':conv2_bias'] = mx.symbol.Variable(self.name + ':conv2_bias')
        return params

    @property
    def dim_out(self):
        return (self.num_filter, self.dim_in[1], self.dim_in[2])

    @property
    def name(self):
        return "ScoreMapProcessor"

    def score_map_processing(self, score_maps, postfix=''):
        score_maps = mx.symbol.Concat(*score_maps, dim=1)
        conv1 = mx.symbol.Convolution(data=score_maps,
                                      weight=self.params[self.name + ':conv1_weight'],
                                      bias=self.params[self.name + ':conv1_bias'],
                                      kernel=(3,3), pad=(1,1),
                                      num_filter=self.num_filter, name=self.name + ':conv1' + postfix)
        act1 = mx.symbol.Activation(data=conv1, act_type='relu', name=self.name + ':act1' + postfix)
        conv2 = mx.symbol.Convolution(data=act1,
                                      weight=self.params[self.name + ':conv2_weight'],
                                      bias=self.params[self.name + ':conv2_bias'],
                                      kernel=(3, 3), pad=(1, 1),
                                      num_filter=self.num_filter, name=self.name + ':conv2' + postfix)
        act2 = mx.symbol.Activation(data=conv2, act_type='relu', name=self.name + ':act2' + postfix)
        return act2

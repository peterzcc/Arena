import mxnet as mx
import numpy
from collections import namedtuple, OrderedDict
from arena.operators import *
from .common import *
from .recurrent import LSTMState, LSTMParam, LSTMLayerProp, step_stack_lstm

'''
ImagePatch stores the basic information of the patch the basic attentional element of the tracker
'''

ImagePatch = namedtuple("ImagePatch", ["center", "size", "data"])

Glimpse = namedtuple("Glimpse", ["center", "size", "data"])


def get_multiscale_size(glimpse):
    size_l = []
    curr_scale = 1.0
    for i in range(glimpse.scale_num):
        size_l.append(glimpse.size * curr_scale)
        curr_scale *= glimpse.scale_mult
    return mx.symbol.Concat(*size_l, num_args=len(size_l), dim=0)

class GlimpseHandler(object):
    def __init__(self, scale_mult, scale_num, output_shape, init_scale=1.0):
        super(GlimpseHandler, self).__init__()
        self.scale_mult = scale_mult
        self.scale_num = scale_num
        self.init_scale = init_scale
        self.output_shape = output_shape

    @property
    def name(self):
        return "GlimpseHandler"

    '''
    pyramid_glimpse: Generate a spatial pyramid of glimpse sectors, pad zero if necessary.
                     Here, center = (cx, cy) and size = (sx, sy)
    '''
    def pyramid_glimpse(self, img, center, size, postfix=''):
        data_l = []
        curr_scale = self.init_scale
        roi = mx.symbol.Concat(*[center, size], num_args=2, dim=1)
        for i in range(self.scale_num):
            patch_data = mx.symbol.SpatialGlimpse(data=img, roi=roi,
                                             output_shape=self.output_shape,
                                             scale=curr_scale)
            data_l.append(patch_data)
            curr_scale *= self.scale_mult
        data = mx.symbol.Concat(*data_l, num_args=len(data_l), dim=0)
        data = mx.symbol.BlockGrad(data, name=self.name + ":glimpse%s" % postfix)
        return Glimpse(center=center, size=size, data=data)


'''
Output the predicted ROI given the anchor ROI and the transformed ROI. The transformaion rule is:

pred_cx = anchor_cx + transformation_cx * anchor_sx
pred_cy = anchor_cy + transformation_cy * anchor_sy
pred_sx = exp(transformation_sx) * anchor_sx
pred_sy = exp(transformation_sy) * anchor_sy

anchor, transformation and truth store the ROIs: (cx, cy, sx, sy)

The gradient is computed with-respect-to the Huber loss:
f(x, x_truth) =
if ||x||_2 < 1:
    0.5 * ||x - x_truth||_2^2
else:
    |x - x_truth| - 0.5
'''

class BoundingBoxRegressionOp(mx.operator.NumpyOp):
    def __init__(self):
        super(BoundingBoxRegressionOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['transformation', 'anchor', 'truth']

    def list_outputs(self):
        return ['loss']

    def infer_shape(self, in_shape):
        transformation_shape = in_shape[0]
        anchor_shape = in_shape[0]
        truth_shape = in_shape[0]
        output_shape = in_shape[0]
        return [transformation_shape, anchor_shape, truth_shape], \
               [output_shape]

    def forward(self, in_data, out_data):
        transformation = in_data[0]
        anchor = in_data[1]
        output = out_data[0]
        print 'transformation', transformation
        print 'anchor', anchor
        output[:, 0] = anchor[:, 0] + transformation[:, 0] * anchor[:, 2]
        output[:, 1] = anchor[:, 1] + transformation[:, 1] * anchor[:, 3]
        output[:, 2] = numpy.exp(transformation[:, 2]) * anchor[:, 2]
        output[:, 3] = numpy.exp(transformation[:, 3]) * anchor[:, 3]

    def backward(self, out_grad, in_data, out_data, in_grad):
        transformation = in_data[0]
        anchor = in_data[1]
        truth = in_data[2]
        grad_transformation = in_grad[0]
        transformed_truth = numpy.zeros(truth.shape, dtype=numpy.float32)
        transformed_truth[:, 0] = (truth[:, 0] - anchor[:, 0]) / anchor[:, 2]
        transformed_truth[:, 1] = (truth[:, 1] - anchor[:, 1]) / anchor[:, 3]
        transformed_truth[:, 2] = numpy.log(truth[:, 2] / anchor[:, 2])
        transformed_truth[:, 3] = numpy.log(truth[:, 3] / anchor[:, 3])
        grad_transformation[:] = numpy.clip(transformation - transformed_truth, -1, 1)


def roi_transform(anchor_center, anchor_size, roi):
    roi_sliced = mx.symbol.SliceChannel(roi, num_outputs=2, axis=1)
    transformed_center = (roi_sliced[0] - anchor_center) / anchor_size
    transformed_size = mx.symbol.log(roi_sliced[1] / anchor_size)
    return transformed_center, transformed_size


def roi_transform_inv(anchor_center, anchor_size, transformed_roi):
    roi_sliced = mx.symbol.SliceChannel(transformed_roi, num_outputs=2, axis=1)
    center = anchor_center + roi_sliced[0] * anchor_size
    size = mx.symbol.exp(roi_sliced[1]) * anchor_size
    return center, size


class AttentionHandler(object):
    def __init__(self, glimpse_handler=None, cf_handler=None, scoremap_processor=None,
                 total_steps=None, lstm_layer_props=None, fixed_variance=True):
        super(AttentionHandler, self).__init__()
        self.glimpse_handler = glimpse_handler
        self.cf_handler = cf_handler
        self.scoremap_processor = scoremap_processor
        self.fixed_variance = fixed_variance
        self.lstm_layer_props = lstm_layer_props
        self.roi_encoding_params = self._init_roi_encoding_params()
        self.lstm_params, self.init_lstm_states = self._init_lstm_params()
        self.roi_policy_params = self._init_roi_policy_params()
        self.total_steps = total_steps

    def _init_roi_encoding_params(self):
        params = OrderedDict()
        prefix = self.name + ':roi_encoding'
        params[prefix + ':fc1'] = FCParam(weight=mx.symbol.Variable(prefix + ':fc1_weight'),
                                          bias=mx.symbol.Variable(prefix + ':fc1_bias'))
        params[prefix + ':fc2'] = FCParam(weight=mx.symbol.Variable(prefix + ':fc2_weight'),
                                          bias=mx.symbol.Variable(prefix + ':fc2_bias'))
        return params

    def init_lstm(self, ctx=get_default_ctx()):
        prefix = self.name
        init_lstm_data = OrderedDict()
        init_lstm_shapes = OrderedDict()
        for i, lstm_layer_prop in enumerate(self.lstm_layer_props):
            init_lstm_shapes[prefix + ':init_lstm%d_c' % i] = (1, lstm_layer_prop.num_hidden)
            init_lstm_shapes[prefix + ':init_lstm%d_h' % i] = (1, lstm_layer_prop.num_hidden)
            init_lstm_data[prefix + ':init_lstm%d_c' % i] = nd.zeros((1, lstm_layer_prop.num_hidden), ctx=ctx)
            init_lstm_data[prefix + ':init_lstm%d_h' % i] = nd.zeros((1, lstm_layer_prop.num_hidden), ctx=ctx)
        return init_lstm_data, init_lstm_shapes

    def _init_lstm_params(self):
        params = OrderedDict()
        init_lstm_states = []
        prefix = self.name
        for i, lstm_layer_prop in enumerate(self.lstm_layer_props):
            params[prefix + ':lstm%d' % i] = \
                LSTMParam(i2h_weight=mx.symbol.Variable(prefix + ':lstm%d_i2h_weight' % i),
                          i2h_bias=mx.symbol.Variable(prefix + ':lstm%d_i2h_bias' % i),
                          h2h_weight=mx.symbol.Variable(prefix + ':lstm%d_h2h_weight' % i),
                        h2h_bias=mx.symbol.Variable(prefix + ':lstm%d_h2h_bias' % i))
            init_lstm_states.append(LSTMState(c=mx.symbol.Variable(prefix + ':init_lstm%d_c' % i),
                                              h=mx.symbol.Variable(prefix + ':init_lstm%d_h' % i)))
        return params, init_lstm_states

    def _init_roi_policy_params(self):
        params = OrderedDict()
        prefix = self.name
        # The search roi (transformed) of the next attention step
        params[prefix + ':search_roi_mean'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':search_roi_mean_weight'),
                    bias=mx.symbol.Variable(prefix + ':search_roi_mean_bias'))
        if not self.fixed_variance:
            params[prefix + ':search_roi_var'] = \
                FCParam(weight=mx.symbol.Variable(prefix + ':search_roi_var_weight'),
                        bias=mx.symbol.Variable(prefix + ':search_roi_var_bias'))

        #The initial search roi (transformed) of the next step
        params[prefix + ':init_roi_mean'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':init_roi_mean_weight'),
                    bias=mx.symbol.Variable(prefix + ':init_roi_mean_bias'))
        if not self.fixed_variance:
            params[prefix + ':init_roi_var'] = \
                FCParam(weight=mx.symbol.Variable(prefix + ':init_roi_var_weight'),
                        bias=mx.symbol.Variable(prefix + ':init_roi_var_bias'))

        #The predicted roi (transformed) of the current timestamp
        params[prefix + ':pred_roi_mean'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':pred_roi_mean_weight'),
                    bias=mx.symbol.Variable(prefix + ':pred_roi_mean_bias'))
        if not self.fixed_variance:
            params[prefix + ':pred_roi_var'] = \
                FCParam(weight=mx.symbol.Variable(prefix + ':pred_roi_var_weight'),
                        bias=mx.symbol.Variable(prefix + ':pred_roi_var_bias'))
        return params

    @property
    def name(self):
        return "AttentionHanlder"

    def roi_encoding(self, center, size):
        prefix = self.name + ':roi_encoding'
        roi = mx.symbol.Concat(center, size, num_args=2, dim=1)
        fc1 = mx.symbol.FullyConnected(data=roi, num_hidden=64,
                                       weight=self.roi_encoding_params[prefix + ':fc1'].weight,
                                       bias=self.roi_encoding_params[prefix + ':fc1'].bias,
                                       name=prefix + ':fc1')
        act1 = mx.symbol.Activation(data=fc1, act_type='relu')
        fc2 = mx.symbol.FullyConnected(data=act1, num_hidden=64,
                                       weight=self.roi_encoding_params[prefix + ':fc2'].weight,
                                       bias=self.roi_encoding_params[prefix + ':fc2'].bias,
                                       name=prefix + ':fc2')
        return fc2

    def roi_policy(self, indata, deterministic=False, roi_var=None, roi_type="init_roi", postfix=''):
        assert roi_type == 'init_roi' or roi_type == 'search_roi' or roi_type == 'pred_roi'
        roi_mean = \
            mx.symbol.FullyConnected(data=indata, num_hidden=4,
                                     name=self.name + ':' + roi_type + '_mean' + postfix,
                                     weight=self.roi_policy_params[
                                         self.name + ':' + roi_type + '_mean'].weight,
                                     bias=self.roi_policy_params[
                                         self.name + ':' + roi_type + '_mean'].bias)
        if not self.fixed_variance:
            assert roi_var is None
            roi_var = \
                mx.symbol.FullyConnected(data=indata, num_hidden=4,
                                         name=self.name + ':' + roi_type + '_var' + postfix,
                                         weight=self.roi_policy_params[
                                             self.name + ':' + roi_type + '_var'].weight,
                                         bias=self.roi_policy_params[
                                             self.name + ':' + roi_type + '_var'].bias)
            roi_var = mx.symbol.Activation(data=roi_var, act_type="softrelu")
        policy_op = LogNormalPolicy(deterministic=deterministic)
        roi = policy_op(mean=roi_mean, var=roi_var,
                        name=self.name + ':' + roi_type + postfix)
        return roi, roi_mean, roi_var

    def attend(self, img, init_center, init_size, multiscale_template,
               memory, ground_truth_roi=None,
               deterministic=False, timestamp=0, roi_var=None):
        memory_code = mx.symbol.Concat(*[state.h for state in memory.states],
                                       num_args=len(memory.states), dim=1)
        tracking_states = self.init_lstm_states
        next_step_init_center = None
        next_step_init_size = None
        pred_center = None
        pred_size = None
        sym_out = OrderedDict()
        init_shapes = OrderedDict()
        search_center = init_center
        search_size = init_size
        for i in range(self.total_steps):
            postfix = '_t%d_step%d' % (timestamp, i)
            glimpse = self.glimpse_handler.pyramid_glimpse(
                img=img, center=search_center, size=search_size, postfix=postfix)
            scoremap = \
                self.cf_handler.get_multiscale_scoremap(multiscale_template=multiscale_template,
                                                        img=img,
                                                        center=search_center,
                                                        size=search_size,
                                                        postfix=postfix)
            processed_scoremap = self.scoremap_processor.scoremap_processing(scoremap, postfix)
            flatten_map = mx.symbol.Flatten(data=processed_scoremap)
            roi_code = self.roi_encoding(glimpse.center, glimpse.size)
            aggregate_input = mx.symbol.Concat(flatten_map, roi_code, memory_code, num_args=3,
                                               dim=1)
            new_states = step_stack_lstm(indata=aggregate_input, prev_states=tracking_states,
                                         lstm_props=self.lstm_layer_props,
                                         params=self.lstm_params.values(),
                                         prefix=self.name + ':', postfix=postfix)
            concat_state = mx.symbol.Concat(*[state.h for state in new_states],
                                            num_args=len(new_states), dim=1)
            if i < self.total_steps - 1:
                search_roi, search_roi_mean, search_roi_var = \
                    self.roi_policy(indata=concat_state, deterministic=deterministic,
                                    roi_type="search_roi", roi_var=roi_var, postfix=postfix)
                sym_out[self.name + ':search_roi' + postfix] = search_roi
                init_shapes[self.name + ':search_roi' + postfix + '_score'] = (1,)

                search_center, search_size = \
                    roi_transform_inv(glimpse.center, glimpse.size, search_roi)
            else:
                next_step_init_roi, next_step_init_roi_mean, next_step_init_roi_var = \
                    self.roi_policy(indata=concat_state, deterministic=deterministic,
                                    roi_type="init_roi", roi_var=roi_var, postfix=postfix)
                sym_out[self.name + ':init_roi' + postfix] = next_step_init_roi
                init_shapes[self.name + ':init_roi' + postfix + '_score'] = (1,)

                next_step_init_center, next_step_init_size = \
                    roi_transform_inv(glimpse.center, glimpse.size, next_step_init_roi)

                pred_roi, pred_roi_mean, pred_roi_var = \
                    self.roi_policy(indata=concat_state, deterministic=deterministic,
                                    roi_type="pred_roi", roi_var=roi_var, postfix=postfix)
                pred_center, pred_size = \
                    roi_transform_inv(glimpse.center, glimpse.size, pred_roi)
                sym_out[self.name + ':pred_roi' + postfix] = pred_roi
                init_shapes[self.name + ':pred_roi' + postfix + '_score'] = (1,)

                bb_regress_op = BoundingBoxRegressionOp()
                if ground_truth_roi is not None:
                    bb_regress_roi = \
                        bb_regress_op(
                            anchor=mx.symbol.Concat(glimpse.center, glimpse.size, num_args=2, dim=1),
                            transformation=pred_roi_mean,
                            truth=ground_truth_roi,
                            name=self.name + ':bb_regress_t%d' % timestamp)
                    sym_out[self.name + ':bb_regress_t%d' % timestamp] = bb_regress_roi
                else:
                    bb_regress_roi = \
                        bb_regress_op(anchor=mx.symbol.Concat(glimpse.center, glimpse.size, num_args=2, dim=1),
                                      transformation=pred_roi_mean,
                                      name=self.name + ':bb_regress_t%d' %timestamp)
                    sym_out[self.name + ':bb_regress_t%d' % timestamp] = bb_regress_roi
                    init_shapes[self.name + ':bb_regress_t%d' % timestamp + '_truth'] = (1,)

            tracking_states = new_states

        return tracking_states, next_step_init_center, next_step_init_size, pred_center, pred_size, \
               sym_out, init_shapes

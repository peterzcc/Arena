import mxnet as mx
import numpy
from collections import namedtuple, OrderedDict
from arena.operators import *
from arena.helpers.tracking import *
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
        center = mx.symbol.BlockGrad(center)
        size = mx.symbol.BlockGrad(size)
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
        return ['output']

    def infer_shape(self, in_shape):
        transformation_shape = in_shape[0]
        anchor_shape = in_shape[1]
        truth_shape = in_shape[0]
        output_shape = in_shape[0]
        return [transformation_shape, anchor_shape, truth_shape], \
               [output_shape]

    def forward(self, in_data, out_data):
        transformation = in_data[0]
        anchor = in_data[1]
        output = out_data[0]
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
        transformed_truth[:, 0] = numpy.nan_to_num((truth[:, 0] - anchor[:, 0]) / anchor[:, 2])
        transformed_truth[:, 1] = numpy.nan_to_num((truth[:, 1] - anchor[:, 1]) / anchor[:, 3])
        #TODO Possible Devision by Zero!
        transformed_truth[:, 2] = numpy.nan_to_num(numpy.log(truth[:, 2] / anchor[:, 2]))
        transformed_truth[:, 3] = numpy.nan_to_num(numpy.log(truth[:, 3] / anchor[:, 3]))
        grad_transformation[:] = numpy.clip(transformation - transformed_truth, -1, 1)


def roi_transform_glimpse(roi, glimpse, glimpse_handler):
    transformed_center, transformed_size = \
        roi_transform(anchor_roi=[glimpse.center,
                                  glimpse.size],
                      roi=roi)
    return transformed_center, transformed_size


def roi_transform_glimpse_inv(transformed_roi, glimpse, glimpse_handler):
    center, size = roi_transform_inv(anchor_roi=[glimpse.center,
                                                 glimpse.size],
                                     transformed_roi=transformed_roi)
    return center, size


class AttentionHandler(object):
    def __init__(self, glimpse_handler=None, cf_handler=None, scoremap_processor=None,
                 memory_handler=None,
                 total_steps=None, lstm_layer_props=None,
                 fixed_center_variance=True,
                 fixed_size_variance=True):
        super(AttentionHandler, self).__init__()
        self.glimpse_handler = glimpse_handler
        self.cf_handler = cf_handler
        self.scoremap_processor = scoremap_processor
        self.memory_handler = memory_handler
        self.lstm_layer_props = lstm_layer_props
        self.fixed_center_variance = fixed_center_variance
        self.fixed_size_variance = fixed_size_variance
        #self.roi_encoding_params = self._init_roi_encoding_params()
        self.lstm_params = self._init_lstm_params()
        self.roi_policy_params = self._init_roi_policy_params()
        self.total_steps = total_steps

    def _init_roi_encoding_params(self):
        params = OrderedDict()
        prefix = self.name + ':roi_encoding'
        params[prefix + ':fc1'] = FCParam(weight=mx.symbol.Variable(prefix + ':fc1_weight'),
                                          bias=mx.symbol.Variable(prefix + ':fc1_bias'))
        #params[prefix + ':fc2'] = FCParam(weight=mx.symbol.Variable(prefix + ':fc2_weight'),
        #                                  bias=mx.symbol.Variable(prefix + ':fc2_bias'))
        return params

    def _init_lstm_params(self):
        params = OrderedDict()
        prefix = self.name
        for i, lstm_layer_prop in enumerate(self.lstm_layer_props):
            params[prefix + ':lstm%d' % i] = \
                LSTMParam(i2h_weight=mx.symbol.Variable(prefix + ':lstm%d_i2h_weight' % i),
                          i2h_bias=mx.symbol.Variable(prefix + ':lstm%d_i2h_bias' % i),
                          h2h_weight=mx.symbol.Variable(prefix + ':lstm%d_h2h_weight' % i),
                        h2h_bias=mx.symbol.Variable(prefix + ':lstm%d_h2h_bias' % i))
        return params

    def _init_roi_policy_params(self):
        params = OrderedDict()
        prefix = self.name
        # The search roi (transformed) of the next attention step
        params[prefix + ':trans_search_roi:fc1'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':trans_search_roi:fc1_weight'),
                    bias=mx.symbol.Variable(prefix + ':trans_search_roi:fc1_bias'))
        params[prefix + ':trans_search_roi:fc2'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':trans_search_roi:fc2_weight'),
                    bias=mx.symbol.Variable(prefix + ':trans_search_roi:fc2_bias'))
        params[prefix + ':trans_search_center:mean'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':trans_search_center:mean_weight'),
                    bias=mx.symbol.Variable(prefix + ':trans_search_center:mean_bias'))
        params[prefix + ':trans_search_size:mean'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':trans_search_size:mean_weight'),
                    bias=mx.symbol.Variable(prefix + ':trans_search_size:mean_bias'))
        if not self.fixed_center_variance:
            params[prefix + ':trans_search_center:var'] = \
                FCParam(weight=mx.symbol.Variable(prefix + ':trans_search_center:var_weight'),
                        bias=mx.symbol.Variable(prefix + ':trans_search_center:var_bias'))
        if not self.fixed_size_variance:
            params[prefix + ':trans_search_size:var'] = \
                FCParam(weight=mx.symbol.Variable(prefix + ':trans_search_size:var_weight'),
                        bias=mx.symbol.Variable(prefix + ':trans_search_size:var_bias'))

        # The predicted roi (transformed) of the current timestamp
        params[prefix + ':trans_pred_roi:fc1'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':trans_pred_roi:fc1_weight'),
                    bias=mx.symbol.Variable(prefix + ':trans_pred_roi:fc1_bias'))
        params[prefix + ':trans_pred_roi:fc2'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':trans_pred_roi:fc2_weight'),
                    bias=mx.symbol.Variable(prefix + ':trans_pred_roi:fc2_bias'))
        params[prefix + ':trans_pred_center:mean'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':trans_pred_center:mean_weight'),
                    bias=mx.symbol.Variable(prefix + ':trans_pred_center:mean_bias'))
        params[prefix + ':trans_pred_size:mean'] = \
            FCParam(weight=mx.symbol.Variable(prefix + ':trans_pred_size:mean_weight'),
                    bias=mx.symbol.Variable(prefix + ':trans_pred_size:mean_bias'))
        return params

    @property
    def name(self):
        return "AttentionHanlder"

    def roi_policy(self, indata, deterministic=False, center_var=None, size_var=None,
                   typ="trans_search", postfix=''):
        assert typ == 'trans_search' or typ == 'trans_pred', "The given typ=%s" %typ
        roi_fc1 = \
            mx.symbol.FullyConnected(data=indata, num_hidden=512,
                                     name=self.name + ':' + typ + '_roi:fc1' + postfix,
                                     weight=self.roi_policy_params[
                                         self.name + ':' + typ + '_roi:fc1'].weight,
                                     bias=self.roi_policy_params[
                                         self.name + ':' + typ + '_roi:fc1'].bias)
        roi_act1 = mx.symbol.Activation(data=roi_fc1, act_type='tanh')
        roi_act1 = mx.sym.Dropout(data=roi_act1, p=0.3)
        roi_fc2 = \
            mx.symbol.FullyConnected(data=roi_act1, num_hidden=512,
                                     name=self.name + ':' + typ + '_roi:fc2' + postfix,
                                     weight=self.roi_policy_params[
                                         self.name + ':' + typ + '_roi:fc2'].weight,
                                     bias=self.roi_policy_params[
                                         self.name + ':' + typ + '_roi:fc2'].bias)
        roi_act2 = mx.symbol.Activation(data=roi_fc2, act_type='tanh')
        roi_act2 = mx.sym.Dropout(data=roi_act2, p=0.3)
        center_mean = \
            mx.symbol.FullyConnected(data=roi_act2, num_hidden=2,
                                     name=self.name + ':' + typ + '_center:mean' + postfix,
                                     weight=self.roi_policy_params[
                                         self.name + ':' + typ + '_center:mean'].weight,
                                     bias=self.roi_policy_params[
                                         self.name + ':' + typ + '_center:mean'].bias)
        center_mean = center_mean / 10
        size_mean = \
            mx.symbol.FullyConnected(data=roi_act2, num_hidden=2,
                                     name=self.name + ':' + typ + '_size:mean' + postfix,
                                     weight=self.roi_policy_params[
                                         self.name + ':' + typ + '_size:mean'].weight,
                                     bias=self.roi_policy_params[
                                         self.name + ':' + typ + '_size:mean'].bias)
        size_mean = size_mean
        if typ is not 'trans_pred':
            if center_var is None:
                center_var = \
                    mx.symbol.FullyConnected(data=roi_act2, num_hidden=2,
                                             name=self.name + ':' + typ + '_center:var' + postfix,
                                             weight=self.roi_policy_params[
                                                 self.name + ':' + typ + '_center:var'].weight,
                                             bias=self.roi_policy_params[
                                                 self.name + ':' + typ + '_center:var'].bias)
                center_var = mx.symbol.Activation(data=center_var, act_type="softrelu")
            if size_var is None:
                size_var = \
                    mx.symbol.FullyConnected(data=roi_act2, num_hidden=2,
                                             name=self.name + ':' + typ + '_size:var' + postfix,
                                             weight=self.roi_policy_params[
                                                 self.name + ':' + typ + '_size:var'].weight,
                                             bias=self.roi_policy_params[
                                                 self.name + ':' + typ + '_size:var'].bias)
                size_var = mx.symbol.Activation(data=size_var, act_type="softrelu")
            #policy_op = LogNormalPolicy(deterministic=deterministic)
            #roi = policy_op(mean=roi_mean, var=roi_var,
            #                name=self.name + ':' + roi_type + postfix)
            center_policy_op = LogNormalPolicy(deterministic=deterministic)
            size_policy_op = LogNormalPolicy(deterministic=deterministic)
            center = center_policy_op(mean=center_mean, var=center_var,
                                      name=self.name + ':' + typ +'_center' + postfix)
            size = size_policy_op(mean=size_mean, var=size_var,
                                  name=self.name + ':' + typ + '_size' + postfix)
            return center/5, size * numpy.log(1.02), center_mean/10, size_mean * numpy.log(1.02), \
                   center_var, size_var
        else:
            center = center_mean
            size = size_mean
            return center/5, size * numpy.log(1.02)


    def attend(self, img, init_search_center, init_search_size,
               memory, ground_truth_roi=None,
               deterministic=False, timestamp=0,
               center_var=None, size_var=None):
        tracking_states = memory.states
        pred_center = None
        pred_size = None
        sym_out = OrderedDict()
        init_shapes = OrderedDict()
        search_center = init_search_center
        search_size = init_search_size

        for i in range(self.total_steps):
            postfix = '_t%d_step%d' % (timestamp, i)
            sym_out[self.name + 'real_search_center' + postfix] = mx.symbol.BlockGrad(search_center)
            sym_out[self.name + 'real_search_size' + postfix] = mx.symbol.BlockGrad(search_size)
            glimpse = self.glimpse_handler.pyramid_glimpse(img=img,
                                                           center=search_center,
                                                           size=search_size,
                                                           postfix=postfix)
            #1. Read template from the memory
            memory, template, read_sym_out, read_init_shapes = \
                self.memory_handler.read(memory=memory, glimpse=glimpse, timestamp=timestamp,
                                         attention_step=i)
            sym_out.update(read_sym_out)
            init_shapes.update(read_init_shapes)
            #sym_out['counter_after_read' + postfix] = memory.status.counter
            #sym_out['visiting_timestamp_after_read' + postfix] = memory.status.visiting_timestamp
            scoremap = \
                self.cf_handler.get_multiscale_scoremap(multiscale_template=template,
                                                        glimpse=glimpse,
                                                        postfix=postfix)
            sym_out[self.name + ':attention_scoremap' + postfix] = scoremap
            processed_scoremap = self.scoremap_processor.scoremap_processing(scoremap, postfix)
            flatten_map = mx.symbol.Reshape(processed_scoremap, target_shape=(1, 0))
            tracking_states = step_stack_lstm(indata=flatten_map, prev_states=tracking_states,
                                         lstm_props=self.lstm_layer_props,
                                         params=self.lstm_params.values(),
                                         prefix=self.name + ':', postfix=postfix)
            concat_state = mx.symbol.Concat(*([state.h for state in tracking_states] + [flatten_map]),
                                            num_args=len(tracking_states) + 1, dim=1)
            if i < self.total_steps - 1:
                trans_search_center, trans_search_size, \
                trans_search_center_mean, trans_search_size_mean, \
                trans_search_center_var, trans_search_size_var = \
                    self.roi_policy(indata=concat_state, deterministic=deterministic,
                                    typ="trans_search", center_var=center_var, size_var=size_var,
                                    postfix=postfix)
                sym_out[self.name + ':trans_search_center' + postfix] = trans_search_center
                sym_out[self.name + ':trans_search_size' + postfix] = trans_search_size
                init_shapes[self.name + ':trans_search_center' + postfix + '_score'] = (1,)
                init_shapes[self.name + ':trans_search_size' + postfix + '_score'] = (1,)

                search_center, search_size = \
                    roi_transform_glimpse_inv(transformed_roi=[trans_search_center, trans_search_size],
                                              glimpse=glimpse,
                                              glimpse_handler=self.glimpse_handler)
            else:
                trans_pred_center, trans_pred_size = \
                    self.roi_policy(indata=concat_state, deterministic=deterministic,
                                    typ="trans_pred", center_var=center_var,
                                    size_var=size_var, postfix=postfix)
                sym_out[self.name + ':trans_pred_center' + postfix] = mx.symbol.BlockGrad(trans_pred_center)
                sym_out[self.name + ':trans_pred_size' + postfix] = mx.symbol.BlockGrad(trans_pred_size)
                if ground_truth_roi is not None:
                    trans_ground_truth_center, trans_ground_truth_size = \
                        roi_transform_glimpse(
                            roi=ground_truth_roi, glimpse=glimpse,
                            glimpse_handler=self.glimpse_handler)
                    #TODO Use the MXNet CPP Version : mx.sym.smooth_l1
                    bb_regress_loss = \
                        mx.symbol.smooth_l1(data=mx.symbol.Concat(trans_pred_center,
                                                                  trans_pred_size,
                                                                  num_args=2, dim=1) -
                                                 mx.symbol.Concat(trans_ground_truth_center,
                                                                  trans_ground_truth_size,
                                                                  num_args=2, dim=1),
                                            scalar=1.0)
                    bb_regress_loss = mx.symbol.MakeLoss(
                        name=self.name + ':bb_regress_loss_t%d' % timestamp,
                        data=bb_regress_loss, grad_scale=1.0)
                    sym_out[self.name + ':bb_regress_loss_t%d' % timestamp] = bb_regress_loss

                pred_center, pred_size = \
                    roi_transform_glimpse_inv(
                        transformed_roi=[trans_pred_center, trans_pred_size],
                        glimpse=glimpse,
                        glimpse_handler=self.glimpse_handler
                    )
        next_step_init_center = pred_center
        next_step_init_size = pred_size
        return memory, next_step_init_center, next_step_init_size, pred_center, pred_size, \
               sym_out, init_shapes

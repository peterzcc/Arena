import mxnet as mx
import numpy
from collections import namedtuple, OrderedDict
from arena.operators import *
from arena.helpers.tracking import *
from .common import *
from .recurrent import LSTMState, LSTMParam, LSTMLayerProp, step_stack_lstm
from .memory import Memory
'''
Glimpse stores the multi-scale image patch the tracker sees. It's the basic attentional element of the tracker
'''

Glimpse = namedtuple("Glimpse", ["center", "size", "data"])


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
                 fixed_size_variance=True,
                 verbose_sym_out=False):
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
        self.verbose_sym_out = verbose_sym_out

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
            mx.symbol.FullyConnected(data=indata, num_hidden=256,
                                     name=self.name + ':' + typ + '_roi:fc1' + postfix,
                                     weight=self.roi_policy_params[
                                         self.name + ':' + typ + '_roi:fc1'].weight,
                                     bias=self.roi_policy_params[
                                         self.name + ':' + typ + '_roi:fc1'].bias)
        roi_act1 = mx.symbol.Activation(data=roi_fc1, act_type='tanh')
        roi_act1 = mx.sym.Dropout(data=roi_act1, p=0.3)
        roi_fc2 = \
            mx.symbol.FullyConnected(data=roi_act1, num_hidden=256,
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
        size_mean = \
            mx.symbol.FullyConnected(data=roi_act2, num_hidden=2,
                                     name=self.name + ':' + typ + '_size:mean' + postfix,
                                     weight=self.roi_policy_params[
                                         self.name + ':' + typ + '_size:mean'].weight,
                                     bias=self.roi_policy_params[
                                         self.name + ':' + typ + '_size:mean'].bias)
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
            #roi = mx.symbol.Custom(mean=roi_mean, var=roi_var, deterministic=deterministic
            #                name=self.name + ':' + roi_type + postfix, op_type='LogNormalPolicy')
            center = mx.symbol.Custom(mean=center_mean, var=center_var, deterministic=deterministic,
                                      name=self.name + ':' + typ +'_center' + postfix, op_type='LogNormalPolicy')
            size = mx.symbol.Custom(mean=size_mean, var=size_var, deterministic=deterministic,
                                    name=self.name + ':' + typ + '_size' + postfix, op_type='LogNormalPolicy')
            return center/5, size * numpy.log(1.02), center_mean/5, size_mean * numpy.log(1.02), \
                   center_var, size_var
        else:
            center = center_mean
            size = size_mean * numpy.log(1.02)
            return center, size


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

            processed_scoremap = self.scoremap_processor.scoremap_processing(scoremap, postfix)
            if self.verbose_sym_out:
                sym_out[self.name + ':attention_scoremap' + postfix] = scoremap
                sym_out[self.name + ':processed_scoremap' + postfix] = mx.symbol.BlockGrad(processed_scoremap)
            flatten_map = mx.symbol.Reshape(processed_scoremap, shape=(1, 0))
            tracking_states = step_stack_lstm(indata=flatten_map, prev_states=tracking_states,
                                         lstm_props=self.lstm_layer_props,
                                         params=self.lstm_params.values(),
                                         prefix=self.name + ':', postfix=postfix)
            memory = Memory(numerators=memory.numerators, denominators=memory.denominators, states=tracking_states,
                            status=memory.status)
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

import mxnet as mx
import mxnet.ndarray as nd
import numpy
from collections import namedtuple, OrderedDict
from tracking import ScaleCFTemplate
from arena.operators import *
from .recurrent import LSTMState, LSTMParam, LSTMLayerProp, step_stack_lstm
from .common import *

'''

Memory --> The basic memory structure
numerators: (num_memory, scale_num * channel of single template, cf_out_rows, cf_out_cols)
denominators: (num_memory, scale_num * channel of single template, cf_out_rows, cf_out_cols)
state: list of LSTMState (Multiple layers)
status: Status like reading times and visiting timestamp of each memory cell
'''

Memory = namedtuple("Memory", ["numerators", "denominators", "states", "status"])

'''
MemoryStat --> The statistical variables of the memory
counter: Counter of the memory, (1, memory_size)
visiting_timestamp: The recorded visiting timestamp of the memory elements, (1, memory_size)
'''

MemoryStat = namedtuple("MemoryStat", ["counter", "visiting_timestamp"])


def memory_to_sym_dict(memory):
    ret = OrderedDict()
    ret[memory.numerators.list_outputs()[0]] = memory.numerators
    ret[memory.denominators.list_outputs()[0]] = memory.denominators
    for state in memory.states:
        ret[state.h.list_outputs()[0]] = state.h
        ret[state.c.list_outputs()[0]] = state.c
    ret[memory.status.counter.list_outputs()[0]] = memory.status.counter
    ret[memory.status.visiting_timestamp.list_outputs()[0]] = memory.status.visiting_timestamp
    return ret

class IncreaseElementOp(mx.operator.NumpyOp):
    def __init__(self):
        super(IncreaseElementOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'index']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        index_shape = in_shape[1]
        output_shape = in_shape[0]
        return [data_shape, index_shape], [output_shape]

    def forward(self, in_data, out_data):
        data = in_data[0]
        index = in_data[1].astype(numpy.int)
        output = out_data[0]
        output[:] = data
        output[numpy.arange(data.shape[0]), index] += 1


'''
Get the operation flags and updated counter + pointer + visiting_timestamp for each memory unit
given the `control_flag`

(1) If the mode is set to be 'read':
control_flag: the memory index to read

(2) If the mode is set to be 'write':
control flag: 0 --> No Change,
              1 --> Update the last chosen indices,
              2 --> Replace the oldest memory element

counter: (memory_size,)
visiting_timestamp: (memory_size,)
control_flag: (1,)
new_counter: (memory_size,)
new_visiting_timestamp: (memory_size,)
flags: (memory_size,)
'''


class MemoryStatUpdateOp(mx.operator.NumpyOp):
    def __init__(self, mode='read'):
        super(MemoryStatUpdateOp, self).__init__(need_top_grad=False)
        self.mode = mode
        assert 'read' == self.mode or 'write' == self.mode

    def list_arguments(self):
        return ['counter', 'visiting_timestamp', 'control_flag']

    def list_outputs(self):
        if 'read' == self.mode:
            return ['new_counter', 'new_visiting_timestamp']
        elif 'write' == self.mode:
            return ['new_counter', 'new_visiting_timestamp', 'flags']
        else:
            raise NotImplementedError

    def infer_shape(self, in_shape):
        counter_shape = in_shape[0]
        visiting_timestamp_shape = in_shape[1]
        control_flag_shape = in_shape[2]
        assert len(in_shape[0]) == 1
        assert in_shape[0] == in_shape[1], "Memory Size of the counter and the visiting timestamp" \
                                           "must be the same."
        assert len(in_shape[2]) == 1 and in_shape[2][0] == 1, "in_shape[2] = %s" %str(in_shape[2])
        if 'read' == self.mode:
            new_counter_shape = in_shape[0]
            new_visiting_timestamp_shape = in_shape[1]
            return [counter_shape, visiting_timestamp_shape, control_flag_shape], \
                   [new_counter_shape, new_visiting_timestamp_shape]
        elif 'write' == self.mode:
            new_counter_shape = in_shape[0]
            new_visiting_timestamp_shape = in_shape[1]
            flags_shape = in_shape[0]
            return [counter_shape, visiting_timestamp_shape, control_flag_shape], \
                   [new_counter_shape, new_visiting_timestamp_shape, flags_shape]
        else:
            raise NotImplementedError

    def forward(self, in_data, out_data):
        counter = in_data[0]
        visiting_timestamp = in_data[1]
        control_flag = in_data[2].astype(numpy.int)
        new_counter = out_data[0]
        new_visiting_timestamp = out_data[1]
        new_counter[:] = counter
        new_visiting_timestamp[:] = visiting_timestamp
        if 'read' == self.mode:
            new_counter[control_flag] += 1
            new_visiting_timestamp[control_flag] = numpy.max(visiting_timestamp) + 1
        elif 'write' == self.mode:
            flags = out_data[2]
            if 0 == control_flag:
                flags[:] = 0
            elif 1 == control_flag:
                flags[:] = 0
                write_ind = numpy.argmax(visiting_timestamp)
                flags[write_ind] = 1
                new_counter[write_ind] += 1
                new_visiting_timestamp[write_ind] = numpy.max(visiting_timestamp) + 1
            elif 2 == control_flag:
                flags[:] = 0
                write_ind = numpy.argmin(visiting_timestamp)
                flags[write_ind] = 2
                new_counter[write_ind] = 1
                new_visiting_timestamp[write_ind] = numpy.max(visiting_timestamp) + 1
            else:
                raise NotImplementedError, \
                    "Control Flag Must be 0, 1 or 2, received %d for control_flags" \
                    % control_flag
        else:
            raise NotImplementedError


class MemoryHandler(object):
    def __init__(self, cf_handler, scoremap_processor, memory_size=4,
                 lstm_layer_props=None):
        self.memory_size = memory_size
        self.lstm_layer_props = lstm_layer_props
        self.cf_handler = cf_handler
        self.scoremap_processor = scoremap_processor
        self.write_params = self._init_write_params()
        self.read_params = self._init_read_params()
        self.state_transition_params = self._init_state_transition_params()

    @property
    def scale_num(self):
        return self.cf_handler.scale_num

    def _init_write_params(self):
        params = OrderedDict()
        prefix = self.name + ':write'
        params[prefix + ':fc1'] = FCParam(weight=mx.symbol.Variable(prefix + ':fc1_weight'),
                                          bias=mx.symbol.Variable(prefix + ':fc1_bias'))
        params[prefix + ':fc2'] = FCParam(weight=mx.symbol.Variable(prefix + ':fc2_weight'),
                                          bias=mx.symbol.Variable(prefix + ':fc2_bias'))
        return params

    def _init_read_params(self):
        params = OrderedDict()
        prefix = self.name + ':read'
        params[prefix + ':fc1'] = FCParam(weight=mx.symbol.Variable(prefix + ':fc1_weight'),
                                          bias=mx.symbol.Variable(prefix + ':fc1_bias'))

        params[prefix + ':fc2'] = FCParam(weight=mx.symbol.Variable(prefix + ':fc2_weight'),
                                          bias=mx.symbol.Variable(prefix + ':fc2_bias'))
        return params

    def _init_state_transition_params(self):
        params = OrderedDict()
        prefix = self.name + ':state_transition'
        for i in range(len(self.lstm_layer_props)):
            # The i-th LSTM layer
            params[prefix + ':lstm%d' % i] = \
                LSTMParam(i2h_weight=mx.symbol.Variable(prefix + ':lstm%d_i2h_weight' % i),
                          i2h_bias=mx.symbol.Variable(prefix + ':lstm%d_i2h_bias' % i),
                          h2h_weight=mx.symbol.Variable(prefix + ':lstm%d_h2h_weight' % i),
                          h2h_bias=mx.symbol.Variable(prefix + ':lstm%d_h2h_bias' % i))
        return params

    def init_memory(self, ctx=get_default_ctx()):
        prefix = self.name + ':memory_init'
        numerators=mx.symbol.Variable(prefix + ':numerators')
        denominators=mx.symbol.Variable(prefix + ':denominators')
        init_memory_status = \
            MemoryStat(counter=mx.symbol.Variable(prefix + ':counter'),
                       visiting_timestamp=mx.symbol.Variable(prefix + ':visiting_timestamp'))
        init_memory_states = []
        for i in range(len(self.lstm_layer_props)):
            init_memory_states.append(
                LSTMState(c=mx.symbol.Variable(prefix + ':lstm%d_c' % i),
                          h=mx.symbol.Variable(prefix + ':lstm%d_h' % i)))

        data_shapes = OrderedDict()
        init_memory_data = OrderedDict()

        data_shapes[prefix + ':numerators'] =\
            (self.memory_size, self.cf_handler.scale_num * self.cf_handler.channel_size,
             self.cf_handler.out_rows, self.cf_handler.out_cols)
        data_shapes[prefix + ':denominators'] = \
            (self.memory_size, self.cf_handler.scale_num * 1,
             self.cf_handler.out_rows, self.cf_handler.out_cols)
        data_shapes[prefix + ':counter'] = (self.memory_size,)
        data_shapes[prefix + ':visiting_timestamp'] = (self.memory_size,)
        for i, prop in enumerate(self.lstm_layer_props):
            data_shapes[prefix + ':lstm%d_c' % i] = (1, prop.num_hidden)
            data_shapes[prefix + ':lstm%d_h' % i] = (1, prop.num_hidden)

        for k, v in data_shapes.items():
            init_memory_data[k] = nd.zeros(shape=v, ctx=ctx)

        return Memory(numerators=numerators,
                      denominators=denominators,
                      states=init_memory_states,
                      status=init_memory_status), init_memory_data, data_shapes

    @property
    def name(self):
        return "MemoryHandler"

    def get_write_control_flag(self, memory, tracking_state, deterministic, timestamp=0):
        prefix = self.name + ':write'
        postfix = '_t%d' % timestamp

        # 1. Update the state of the memory
        assert len(memory.states) == len(self.lstm_layer_props)
        new_memory_states = step_stack_lstm(indata=tracking_state, prev_states=memory.states,
                                            lstm_props=self.lstm_layer_props,
                                            params=self.state_transition_params.values(),
                                            prefix=self.name + ':state_transition',
                                            postfix=postfix)
        # 2. Choose the control flag
        fc1 = mx.symbol.FullyConnected(
            data=mx.symbol.Concat(*[state.h for state in new_memory_states],
                                  num_args=len(new_memory_states), dim=1),
            num_hidden=128,
            weight=self.write_params[prefix + ':fc1'].weight,
            bias=self.write_params[prefix + ':fc1'].bias,
            name=prefix + ':fc1' + postfix)
        act1 = mx.symbol.Activation(data=fc1, act_type='relu', name=prefix + ':relu1' + postfix)
        fc2 = mx.symbol.FullyConnected(data=act1,
                                       num_hidden=3,
                                       weight=self.write_params[prefix + ':fc2'].weight,
                                       bias=self.write_params[prefix + ':fc2'].bias,
                                       name=prefix + ':fc2' + postfix)
        control_flag_policy_op = LogSoftmaxPolicy(deterministic=deterministic)
        control_flag = control_flag_policy_op(data=fc2, name=prefix + ':control_flag' + postfix)
        # TODO Enable actor-critic (Like the Async RL paper)
        return control_flag, new_memory_states

    def write(self, memory, update_multiscale_template, control_flag=None, update_factor=None,
              tracking_state=None,
              timestamp=0, deterministic=False):
        assert update_factor is not None, 'Automatic inference of the update_factor is not supported currently!'
        prefix = self.name + ':write'
        postfix = '_t%d' % timestamp
        new_memory_states = memory.states
        # 1. Choose the control flag
        sym_out = OrderedDict()
        init_shapes = OrderedDict()

        if control_flag is None:
            assert tracking_state is not None, "Tracking state must be set for automatic control!"
            control_flag, new_memory_states = \
                self.get_write_control_flag(memory=memory, tracking_state=tracking_state,
                                            deterministic=deterministic, timestamp=timestamp)
            sym_out[prefix + ':control_flag' + postfix + '_action'] = control_flag[0]
            sym_out[prefix + ':control_flag' + postfix + '_prob'] = control_flag[1]
            init_shapes[prefix + ':control_flag' + postfix + '_score'] = (1,)
            control_flag = mx.symbol.Reshape(control_flag[0], target_shape=(0,))
            control_flag = mx.symbol.BlockGrad(control_flag)
        # 2. Update the memory status
        # TODO Change the updating logic (Train the update factor using Reinforcement Unit like Beta-Policy)
        memory_write_control_op = MemoryStatUpdateOp(mode='write')
        new_status = memory_write_control_op(counter=memory.status.counter,
                                             visiting_timestamp=memory.status.visiting_timestamp,
                                             control_flag=control_flag,
                                             name=prefix + ':status' + postfix)
        flag = new_status[2]
        new_status = MemoryStat(new_status[0], new_status[1])

        # 3. Update the multiscale templates
        update_numerator = self.reshape_to_memory_ele(update_multiscale_template.numerator)
        update_denominator = self.reshape_to_memory_ele(update_multiscale_template.denominator)

        new_numerators = mx.symbol.MemoryUpdate(data=memory.numerators,
                                                update=update_numerator,
                                                flag=flag,
                                                factor=update_factor,
                                                name=prefix + ':numerators' + postfix)
        new_denominators = mx.symbol.MemoryUpdate(data=memory.denominators,
                                                  update=update_denominator,
                                                  flag=flag,
                                                  factor=update_factor,
                                                  name=prefix + ':denominators' + postfix)

        new_memory = Memory(numerators=new_numerators,
                            denominators=new_denominators,
                            states=new_memory_states,
                            status=new_status)

        return new_memory, sym_out, init_shapes

    def get_read_control_flag(self, memory, img, center, size, deterministic, timestamp=0):
        prefix = self.name + ':read'
        numerators = mx.symbol.SliceChannel(memory.numerators, num_outputs=self.memory_size, axis=0)
        denominators = mx.symbol.SliceChannel(memory.denominators, num_outputs=self.memory_size, axis=0)
        memory_state_code = mx.symbol.Concat(*[state.h for state in memory.states],
                                             num_args=len(memory.states), dim=1)
        # 1. Calculate the matching score of all the memory units to the new glimpse.
        #    Also, get the feature maps based on the scoremaps.
        feature_map_l = []
        for m in range(self.memory_size):
            postfix = "_m%d_t%d" % (m, timestamp)
            numerator = self.reshape_to_cf(numerators[m])
            denominator = self.reshape_to_cf(denominators[m])
            scoremap = self.cf_handler.get_multiscale_scoremap(
                multiscale_template=ScaleCFTemplate(numerator=numerator, denominator=denominator),
                img=img,
                center=center,
                size=size,
                postfix=postfix)
            feature_map = self.scoremap_processor.scoremap_processing(scoremap, postfix)
            feature_map_l.append(feature_map)
        feature_maps = mx.symbol.Concat(*feature_map_l, num_args=self.memory_size, dim=0)
        feature_maps = mx.symbol.BlockGrad(feature_maps)
        # 2. Compute the scores: Shape --> (1, memory_size)
        postfix = "_t%d" % timestamp
        global_pooled_feature = mx.symbol.Pooling(data=feature_maps,
                                                  kernel=(self.scoremap_processor.dim_out[1],
                                                          self.scoremap_processor.dim_out[2]),
                                                  pool_type="avg",
                                                  name=prefix + ":global-pooling" + postfix)
        global_pooled_feature = mx.symbol.Reshape(data=global_pooled_feature,
                                                  target_shape=(self.memory_size, 0))

        # Here we concatenate the global pooled features to the memory state
        memory_state_code = mx.symbol.Concat(*[memory_state_code for i in range(self.memory_size)],
                                             num_args=self.memory_size,
                                             dim=0)
        concat_feature = mx.symbol.Concat(global_pooled_feature, memory_state_code,
                                          num_args=2, dim=1)
        fc1 = mx.symbol.FullyConnected(data=concat_feature,
                                       num_hidden=256,
                                       weight=self.read_params[prefix + ':fc1'].weight,
                                       bias=self.read_params[prefix + ':fc1'].bias,
                                       name=prefix + ':fc1' + postfix)
        act1 = mx.symbol.Activation(data=fc1, act_type='relu', name=prefix + ':relu1' + postfix)
        fc2 = mx.symbol.FullyConnected(data=act1,
                                       num_hidden=1,
                                       weight=self.read_params[prefix + ':fc2'].weight,
                                       bias=self.read_params[prefix + ':fc2'].bias,
                                       name=prefix + ':fc2' + postfix)
        # We need to swapaxis here since the shape fc2 is (memory_size, 1)
        score = mx.symbol.SwapAxis(fc2, dim1=0, dim2=1, name=prefix + ':score' + postfix)

        # 3. Choose the memory indices based on the computed score and the memory status
        choice_policy_op = LogSoftmaxMaskPolicy(deterministic=deterministic)
        chosen_ind = choice_policy_op(data=score,
                                      mask=mx.symbol.Reshape(memory.status.counter, target_shape=(1, 0)),
                                      name=prefix + ':chosen_ind' + postfix)
        return chosen_ind

    def reshape_to_cf(self, memory_ele_sym, name=None):
        if name is None:
            return mx.symbol.Reshape(memory_ele_sym, target_shape=(self.scale_num, 0,
                                                    self.cf_handler.out_rows,
                                                    self.cf_handler.out_cols))
        else:
            return mx.symbol.Reshape(memory_ele_sym, name=name, target_shape=(self.scale_num, 0,
                                                            self.cf_handler.out_rows,
                                                            self.cf_handler.out_cols))

    def reshape_to_memory_ele(self, cf_sym, name=None):
        if name is None:
            return mx.symbol.Reshape(cf_sym, target_shape=(1, 0,
                                                    self.cf_handler.out_rows,
                                                    self.cf_handler.out_cols))
        else:
            return mx.symbol.Reshape(cf_sym, name=name, target_shape=(1, 0,
                                                    self.cf_handler.out_rows,
                                                    self.cf_handler.out_cols))

    def read(self, memory, img, center, size, chosen_ind=None, deterministic=False, timestamp=0):
        prefix = self.name + ':read'
        postfix = "_t%d" % timestamp
        sym_out = OrderedDict()
        init_shapes = OrderedDict()
        if chosen_ind is None:
            chosen_ind = self.get_read_control_flag(memory=memory,
                                                    img=img,
                                                    center=center,
                                                    size=size,
                                                    timestamp=timestamp,
                                                    deterministic=deterministic)
            sym_out[prefix + ':chosen_ind' + postfix + '_action'] = chosen_ind[0]
            sym_out[prefix + ':chosen_ind' + postfix + '_prob'] = chosen_ind[1]
            init_shapes[prefix + ':chosen_ind' + postfix + '_score'] = (1,)
            chosen_ind = chosen_ind[0]

        # 3. Update the memory status
        memory_read_control_op = MemoryStatUpdateOp(mode='read')
        new_status = memory_read_control_op(counter=memory.status.counter,
                                            visiting_timestamp=memory.status.visiting_timestamp,
                                            control_flag=chosen_ind,
                                            name=prefix + ':status' + postfix)
        new_status = MemoryStat(new_status[0], new_status[1])
        new_memory = Memory(numerators=memory.numerators,
                            denominators=memory.denominators,
                            states=memory.states,
                            status=new_status)

        # 4. Choose the memory element
        chosen_numerator = mx.symbol.MemoryChoose(data=memory.numerators, index=chosen_ind)
        chosen_denominator = mx.symbol.MemoryChoose(data=memory.denominators, index=chosen_ind)
        chosen_numerator = self.reshape_to_cf(chosen_numerator,
                                              name=prefix + ":chosen_numerator" + postfix)
        chosen_denominator = self.reshape_to_cf(chosen_denominator,
                                                name=prefix + ":chosen_denominator" + postfix)
        chosen_multiscale_template = ScaleCFTemplate(numerator=chosen_numerator,
                                                     denominator=chosen_denominator)
        return new_memory, chosen_multiscale_template, sym_out, init_shapes

import mxnet as mx
import numpy
from collections import namedtuple, OrderedDict
from tracking import ScaleCFTemplate
from arena.operators import *
from .recurrent import LSTMState, LSTMParam, LSTMLayerProp, step_stack_lstm
from .common import *

'''

Memory --> The basic memory structure
numerators: (num_memory, scale_num * channel of single template, row, col)
denominators: (num_memory, scale_num * channel of single template, row, col)
state: list of LSTMState (Multiple layers)
status: Status like reading times and visiting timestamp of each memory cell

MemoryStat --> The statistical variables of the memory
counter: Counter of the memory
visiting_timestamp: The recorded visiting timestamp of the memory elements
'''

Memory = namedtuple("Memory", ["numerators", "denominators", "states", "status"])

MemoryStat = namedtuple("MemoryStat", ["counter", "visiting_timestamp"])


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
        assert (counter.shape[0] == visiting_timestamp.shape[0]) and \
               (
               visiting_timestamp.shape[0] == control_flag.shape[0]), "Batchsize of all inputs must" \
                                                                      "be the same."
        assert visiting_timestamp.shape == counter.shape
        assert 1 == control_flag.ndim
        assert 2 == counter.ndim
        new_counter = out_data[0]
        new_visiting_timestamp = out_data[1]
        new_counter[:] = counter
        new_visiting_timestamp[:] = visiting_timestamp
        if 'read' == self.mode:
            new_counter[numpy.arange(counter.shape[0]), control_flag] += 1
            new_visiting_timestamp[numpy.arange(visiting_timestamp.shape[0]), control_flag] = \
                numpy.max(visiting_timestamp, axis=1) + 1
        elif 'write' == self.mode:
            flags = out_data[2]
            for i in range(counter.shape[0]):
                if 0 == control_flag[i]:
                    flags[:] = 0
                elif 1 == control_flag[i]:
                    flags[:] = 0
                    flags[i, numpy.argmax(visiting_timestamp[i])] = 1
                elif 2 == control_flag[i]:
                    flags[:] = 0
                    write_ind = numpy.argmin(visiting_timestamp[i])
                    flags[i, write_ind] = 2
                    new_counter[i, write_ind] = 1
                    new_visiting_timestamp[i, write_ind] = numpy.max(visiting_timestamp[i]) + 1
                else:
                    raise NotImplementedError, \
                        "Control Flag Must be 0, 1 or 2, received %d for control_flags[%d]" \
                        % (control_flag[i], i)
        else:
            raise NotImplementedError


class MemoryHandler(object):
    def __init__(self, cf_handler, scoremap_processor, scale_num=1, memory_size=4,
                 lstm_layer_props=None):
        self.scale_num = scale_num
        self.memory_size = memory_size
        self.lstm_layer_props = lstm_layer_props
        self.cf_handler = cf_handler
        self.scoremap_processor = scoremap_processor
        self.write_params = self._init_write_params()
        self.read_params = self._init_read_params()
        self.state_transition_params, self.init_memory_state = self._init_state_transition_params()
        self.update_factor = mx.symbol.Variable(self.name + ':update_factor')

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
        init_memory_state = []
        prefix = self.name + ':state_transition'
        for i in range(len(self.lstm_layer_props)):
            # The i-th LSTM layer
            params[prefix + ':lstm%d' % i] = \
                LSTMParam(i2h_weight=mx.symbol.Variable(prefix + ':lstm%d_i2h_weight' % i),
                          i2h_bias=mx.symbol.Variable(prefix + ':lstm%d_i2h_bias' % i),
                          h2h_weight=mx.symbol.Variable(prefix + ':lstm%d_h2h_weight' % i),
                          h2h_bias=mx.symbol.Variable(prefix + ':lstm%d_h2h_bias' % i))
            init_memory_state.append(
                LSTMState(c=mx.symbol.Variable(prefix + ':init_lstm%d_c' % i),
                          h=mx.symbol.Variable(prefix + ':init_lstm%d_h' % i)))
        return params, init_memory_state

    def init_memory(self):
        prefix = self.name + ':memory_init'
        numerators=mx.symbol.Variable(prefix + ':numerators')
        denominators=mx.symbol.Variable(prefix + ':denominators')
        init_memory_status = \
            MemoryStat(counter=mx.symbol.Variable(prefix + ':counter'),
                       visiting_timestamp=mx.symbol.Variable(prefix + ':visiting_timestamp'))
        return Memory(numerators=numerators,
                      denominators=denominators,
                      state=self.init_memory_state,
                      status=init_memory_status)

    @property
    def name(self):
        return "MemoryHandler"

    def write(self, memory, update_multiscale_template, tracking_state,
              timestamp=0, deterministic=False):
        prefix = self.name + ':write'
        postfix = '_t%d' % timestamp

        rl_sym_out = OrderedDict()
        # 1. Update the state of the memory
        assert len(memory.states) == len(self.lstm_layer_props)
        new_memory_states = step_stack_lstm(indata=tracking_state, prev_states=memory.states,
                                            lstm_props=self.lstm_layer_props,
                                            params=self.state_transition_params.values(),
                                            prefix=self.name + '-state:',
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
        control_flag = control_flag_policy_op(data=fc2)
        rl_sym_out[self.name + ':control_flag' + postfix] = control_flag

        #TODO Change the updating logic (Train the update factor using Reinforcement Unit like Beta-Policy)
        #TODO Enable actor-critic (Like the Async RL paper)


        # 3. Update the memory status
        memory_write_control_op = MemoryStatUpdateOp(mode='write')
        new_status = memory_write_control_op(counter=memory.stat.counter,
                                             visiting_timestamp=memory.stat.visiting_timestamp,
                                             control_flag=control_flag[0],
                                             name="memory_write_stat_t%d" % timestamp)
        flag = new_status[2]
        new_status = MemoryStat(new_status[0], new_status[1])

        # 4. Update the multiscale templates
        update_numerator = mx.symbol.SliceChannel(update_multiscale_template.numerator,
                                                  num_outputs=self.scale_num, axis=0)
        update_numerator = mx.symbol.Concat(update_numerator, num_args=self.scale_num, dim=1)
        update_denominator = mx.symbol.SliceChannel(update_multiscale_template.denominator,
                                                    num_outputs=self.scale_num, axis=0)
        update_denominator = mx.symbol.Concat(update_denominator, num_args=self.scale_num, dim=1)
        new_numerators = mx.symbol.MemoryUpdate(data=memory.numerators,
                                                update=update_numerator,
                                                flag=flag,
                                                factor=self.update_factor)
        new_denominators = mx.symbol.MemoryUpdate(data=memory.denominators,
                                                  update=update_denominator,
                                                  flag=flag,
                                                  factor=self.update_factor)

        new_memory = Memory(numerators=new_numerators,
                            denominators=new_denominators,
                            states=new_memory_states,
                            status=new_status)

        return new_memory, rl_sym_out

    def read(self, memory, glimpse, timestamp=0, deterministic=False):
        prefix = self.name + '-read'
        rl_sym_out = OrderedDict()
        # 1. Calculate the matching score of all the memory units to the new glimpse
        score_l = []
        numerators = mx.symbol.SliceChannel(memory.numerators, num_outputs=self.memory_size, axis=0)
        denominators = mx.symbol.SliceChannel(memory.denominators, num_outputs=self.memory_size, axis=0)
        memory_state_code = mx.symbol.Concat(*[state.h for state in memory.states],
                                             num_args=len(memory.states), dim=1)
        for i in range(self.memory_size):
            postfix = "_m%d_t%d" % (i, timestamp)
            scoremap = self.cf_handler.get_multiscale_scoremap(
                ScaleCFTemplate(numemartor=numerators[i], denominator=denominators[i]),
                glimpse, postfix=postfix)
            feature_map = self.scoremap_processor.scoremap_processing(scoremap, postfix)
            global_pooled_feature = mx.symbol.Pooling(data=feature_map,
                                                      kernel=(self.scoremap_processor.dim_out[1],
                                                              self.scoremap_processor.dim_out[2]),
                                                      pool_type="avg",
                                                      name=prefix + ":global-pooling" + postfix)
            # Here we concatenate the global pooled features to the memory state
            concat_feature = mx.symbol.Concat(global_pooled_feature, memory_state_code, dim=1)
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
            score_l.append(fc2)
        postfix = "_t%d" % timestamp
        score = mx.symbol.Concat(*score_l, dim=1, name=prefix + ':score' + postfix)

        # 2. Choose the memory indices based on the computed score and the memory status
        choice_policy_op = LogSoftmaxMaskPolicy(deterministic=deterministic)
        chosen_ind = choice_policy_op(data=score, mask=memory.stat.counter,
                                      name=prefix + ':chosen_ind' + postfix)
        rl_sym_out[prefix + ':chosen_ind' + postfix] = chosen_ind

        # 3. Update the memory status
        memory_read_control_op = MemoryStatUpdateOp(mode='read')
        new_status = memory_read_control_op(counter=memory.stat.counter,
                                            visiting_timestamp=memory.stat.visiting_timestamp,
                                            control_flag=chosen_ind[0],
                                            name="memory_read_stat_t%d" % timestamp)
        new_status = MemoryStat(new_status[0], new_status[1])
        new_memory = Memory(numerators=memory.numerators,
                            denominators=memory.denominators,
                            states=memory.states,
                            status=new_status)

        # 4. Choose the memory element
        chosen_numerator = mx.symbol.MemoryChoose(data=memory.numerators, index=chosen_ind[0])
        chosen_denominator = mx.symbol.MemoryChoose(data=memory.denominators, index=chosen_ind[0])
        chosen_multiscale_template = ScaleCFTemplate(numerator=chosen_numerator,
                                                     denominator=chosen_denominator)
        return new_memory, chosen_multiscale_template, rl_sym_out

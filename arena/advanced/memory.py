import mxnet as mx
import numpy
from collections import namedtuple

Memory = namedtuple("Memory", ["numerators", "denominators", "stat"])

MemoryElement = namedtuple("MemoryElement", ["numerators", "denominators"])

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
Get the operation flags and updated counter + pointer + visiting_timestamp for each memory unit given the
`control_flag` and

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
               (visiting_timestamp.shape[0] == control_flag.shape[0]), "Batchsize of all inputs must" \
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
            new_visiting_timestamp[numpy.arange(visiting_timestamp.shape[0]), control_flag] =\
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
'''
Function: memory_read
Description: Read from the memory
'''
def memory_read(memory, ind, timestamp=0):
    numerator_chosen = mx.symbol.MemoryChoose(data=memory.numerators, index=ind,
                                              name="memory_read_numerator_t%d" %timestamp)
    denominator_chosen = mx.symbol.MemoryChoose(data=memory.denominators, index=ind,
                                                name="memory_read_denominator_t%d" % timestamp)
    memory_read_control_op = MemoryStatUpdateOp(mode='read')
    l = memory_read_control_op(counter=memory.counter,
                              visiting_timestamp=memory.visiting_timestamp,
                              control_flag=memory.control_flag,
                              name="memory_read_stat_t%d" %timestamp)
    new_stat = MemoryStat(l[0], l[1])
    return MemoryElement(numerator=numerator_chosen, denominator=denominator_chosen), \
           Memory(numerators=memory.numerators, denominators=memory.denominators,
                  stat=new_stat)

'''
Function: memory_write
Description: Update the memory content
'''
def memory_write(memory, memory_element, control_flag, update_factor, timestamp=0):
    new_numerators = []
    new_denominators = []
    # 1. Update numerators and denominators
    memory_write_control_op = MemoryStatUpdateOp(mode='write')
    l = memory_write_control_op(counter=memory.counter, pointer=memory.pointer,
                                control_flag=control_flag,
                                                 name="memory_write_control_status_t%d" % timestamp)
    flags = mx.symbol.SliceChannel(flags, num_outputs=len(memory.numerators), axis=0)
    for i, (numerator, denominator, flag) in enumerate(zip(memory.numerators, memory.denominators, flags)):
        updated_numerator = mx.symbol.MemoryElementUpdate(data=numerator, update_data=memory_element,
                                                          control_flag=control_flag,
                                                          update_factor=update_factor,
                                                          name="memory_write_numerator_t%d" %timestamp)
        updated_denominator = mx.symbol.MemoryElementUpdate(data=denominator, flag=flag,
                                                            update_factor=update_factor,
                                                            name="memory_write_denominator_t%d" % timestamp)
        new_numerators.append(updated_numerator)
        new_denominators.append(updated_denominator)
    return Memory(numerators=new_numerators, denominators=new_denominators,
                  counter=new_counter, pointer=new_pointer)


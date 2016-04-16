import mxnet as mx
import numpy
from collections import namedtuple

Memory = namedtuple("Memory", ["numerators", "denominators", "counter", "pointer"])

MemoryElement = namedtuple("MemoryElement", ["numerator", "denominator"])

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
        index = in_data[1]
        output = out_data[0]
        output[:] = data
        output[index] += 1

class ClearElementOp(mx.operator.NumpyOp):
    def __init__(self):
        super(ClearElementOp, self).__init__(need_top_grad=False)

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
        index = in_data[1]
        output = out_data[0]
        output[:] = data
        output[index] = 0

class ChooseElementOp(mx.operator.NumpyOp):
    def __init__(self):
        super(ChooseElementOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'index']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        index_shape = in_shape[1]
        output_shape = in_shape[1]
        return [data_shape, index_shape], [output_shape]

    def forward(self, in_data, out_data):
        data = in_data[0]
        index = in_data[1]
        output = out_data[0]
        output[:] = data[index]

'''
Get the operation flags and updated counter + pointer for each memory unit given the
`counter`, `pointer` and `control_flag`

control flag: 0 --> No Change,
              1 --> Update the last chosen indices,
              2 --> Replace the least-used memory element
'''
class MemoryControlOp(mx.operator.NumpyOp):
    def __init__(self):
        super(MemoryControlOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['counter', 'pointer', 'control_flag']

    def list_outputs(self):
        return ['flags', 'new_counter', 'new_pointer']

    def infer_shape(self, in_shape):
        counter_shape = in_shape[0]
        pointer_shape = in_shape[1]
        control_flag_shape = in_shape[2]
        flags_shape = in_shape[0]
        new_counter_shape = in_shape[0]
        new_pointer_shape = in_shape[1]
        return [counter_shape, pointer_shape, control_flag_shape], \
               [flags_shape, new_counter_shape, new_pointer_shape]

    def forward(self, in_data, out_data):
        counter = in_data[0]
        pointer = in_data[1]
        control_flag = in_data[2]
        flags = out_data[0]
        new_counter = out_data[1]
        new_pointer = out_data[2]
        flags[:] = 0
        new_counter[:] = counter
        new_pointer[:] = pointer
        if 1 == control_flag :
            flags[pointer] = 1
        elif 2 == control_flag:
            new_pointer[:] = numpy.argmin(counter)
            flags[new_pointer] = 2
            new_counter[new_pointer] = 1
        assert control_flag == 0 or control_flag == 1 or control_flag == 2, \
            "Control Flag Must be 0, 1 or 2, received %d" % control_flag

'''
Function: memory_read
Description: Read from the memory
'''
def memory_read(memory, ind, timestamp=0):
    numerator_chosen = mx.symbol.MemoryChoose(*(memory.numerators), index=ind,
                                              name="memory_read_numerator_t%d" %timestamp)
    denominator_chosen = mx.symbol.MemoryChoose(*(memory.denominators), index=ind,
                                                name="memory_read_denominator_t%d" % timestamp)
    increase_element_op = IncreaseElementOp()
    new_counter = increase_element_op(data=memory.counter, index=ind,
                                      name="memory_read_counter_t%d" %timestamp)
    return MemoryElement(numerator=numerator_chosen, denominator=denominator_chosen), \
           Memory(numerators=memory.numerators, denominators=memory.denominators,
                  counter=new_counter, pointer=ind)

'''
Function: memory_write
Description: Update the memory content
'''
def memory_write(memory, memory_element, control_flag, update_factor, timestamp=0):
    new_numerators = []
    new_denominators = []
    # 1. Update numerators and denominators
    control_op = MemoryControlOp()
    flags, new_counter, new_pointer = control_op(counter=memory.counter, pointer=memory.pointer,
                                                 control_flag=control_flag,
                                                 name="memory_write_control_status_t%d" % timestamp)
    flags = mx.symbol.SliceChannel(flags, num_outputs=len(memory.numerators), axis=0)
    for i, (numerator, denominator, flag) in enumerate(zip(memory.numerators, memory.denominators, flags)):
        updated_numerator = mx.symbol.MemoryElementUpdate(data=numerator, flag=flag,
                                                          update_factor=update_factor,
                                                          name="memory_write_numerator_t%d" %timestamp)
        updated_denominator = mx.symbol.MemoryElementUpdate(data=denominator, flag=flag,
                                                            update_factor=update_factor,
                                                            name="memory_write_denominator_t%d" % timestamp)
        new_numerators.append(updated_numerator)
        new_denominators.append(updated_denominator)
    return Memory(numerators=new_numerators, denominators=new_denominators,
                  counter=new_counter, pointer=new_pointer)


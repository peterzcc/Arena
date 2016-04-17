import mxnet as mx
import numpy
import cv2
from arena import Base
import time
import pyfftw
from arena.advanced.memory import IncreaseElementOp, MemoryStatUpdateOp

def memory_stat_read_write():
    counter = mx.symbol.Variable('counter')
    visiting_timestamp = mx.symbol.Variable('visiting_timestamp')
    control_flag = mx.symbol.Variable('control_flag')
    memory_write_control_op = MemoryStatUpdateOp(mode='write')
    memory_read_control_op = MemoryStatUpdateOp(mode='read')
    controlling_stats_afterwrite = memory_write_control_op(counter=counter, visiting_timestamp=visiting_timestamp,
                                                        control_flag=control_flag)
    controlling_stats_afterread = memory_read_control_op(counter=counter, visiting_timestamp=visiting_timestamp,
                                                control_flag=control_flag)

    data_shapes = {'counter': (1, 4), 'visiting_timestamp': (1, 4), 'control_flag':(1,)}

    write_net = Base(sym=controlling_stats_afterwrite, data_shapes=data_shapes, name='write_net')
    read_net = Base(sym=controlling_stats_afterread, data_shapes=data_shapes, name='read_net')

    current_counter = numpy.array([[10, 20, 3, 40]])
    current_visiting_timestamp = numpy.array([[1, 3, 2, 4]])

    for i in range(100):
        write_outputs = write_net.forward(data_shapes=data_shapes, counter=current_counter,
                              visiting_timestamp=current_visiting_timestamp, control_flag=numpy.array([i%3,]))
        current_counter = write_outputs[0].asnumpy()
        current_visiting_timestamp = write_outputs[1].asnumpy()
        print 'Control Flag:', i%3, 'Counter:', current_counter, \
            " Visiting Timestamp:", current_visiting_timestamp, "Write Ind:", write_outputs[2].asnumpy()

        read_outputs = read_net.forward(data_shapes=data_shapes, counter=current_counter,
                              visiting_timestamp=current_visiting_timestamp, control_flag=numpy.array([i%4,]))
        current_counter = read_outputs[0].asnumpy()
        current_visiting_timestamp = read_outputs[1].asnumpy()
        print 'Control Flag:', i%4, 'Counter:', current_counter, \
            " Visiting Timestamp:", current_visiting_timestamp
        ch = raw_input()

def memory_choose_test():
    memory = mx.symbol.Variable('memory')
    index = mx.symbol.Variable('index')
    chosen_unit = mx.symbol.MemoryChoose(data=memory, index=index)
    data_shapes ={'memory': (5, 4, 3, 3), 'index': (1,)}
    net = Base(sym=chosen_unit, data_shapes=data_shapes)
    memory_npy = numpy.zeros((5, 4, 3, 3), dtype=numpy.float32)
    for i in range(5):
        memory_npy[i, :, :, :] = i
    index_npy = numpy.array([3], dtype=numpy.float32)
    print net.internal_sym_names

    output = net.forward(data_shapes=data_shapes, memory=memory_npy, index=index_npy)[0].asnumpy()
    print output
    print output.shape


memory_choose_test()
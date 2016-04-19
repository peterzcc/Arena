import mxnet as mx
import numpy
import cv2
from arena import Base
import time
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
        write_net.backward(data_shapes=data_shapes)
        current_counter = write_outputs[0].asnumpy()
        current_visiting_timestamp = write_outputs[1].asnumpy()
        print 'Control Flag:', i%3, 'Counter:', current_counter, \
            " Visiting Timestamp:", current_visiting_timestamp, "Flags:", write_outputs[2].asnumpy()

        read_outputs = read_net.forward(data_shapes=data_shapes, counter=current_counter,
                              visiting_timestamp=current_visiting_timestamp, control_flag=numpy.array([i%4,]))
        read_net.backward(data_shapes=data_shapes)
        current_counter = read_outputs[0].asnumpy()
        current_visiting_timestamp = read_outputs[1].asnumpy()
        print 'Control Flag:', i%4, 'Counter:', current_counter, \
            " Visiting Timestamp:", current_visiting_timestamp
        ch = raw_input()

def memory_choose_test():
    memory = mx.symbol.Variable('memory')
    index = mx.symbol.Variable('index')
    chosen_unit = mx.symbol.MemoryChoose(data=memory, index=index)
    chosen_unit = mx.symbol.BlockGrad(data=chosen_unit)
    data_shapes ={'memory': (5, 4, 3, 3), 'index': (1,)}
    net = Base(sym=chosen_unit, data_shapes=data_shapes)
    memory_npy = numpy.zeros((5, 4, 3, 3), dtype=numpy.float32)
    for i in range(5):
        memory_npy[i, :, :, :] = i
    index_npy = numpy.array([3], dtype=numpy.float32)
    print net.internal_sym_names

    output = net.forward(data_shapes=data_shapes, memory=memory_npy, index=index_npy)[0].asnumpy()
    net.backward(data_shapes=data_shapes, memory=memory_npy, index=index_npy)
    print output
    print output.shape

def memory_update_test():
    memory = mx.symbol.Variable('memory')
    update = mx.symbol.Variable('update')
    flag = mx.symbol.Variable('flag')
    update_factor = mx.symbol.Variable('update_factor')
    output = mx.symbol.MemoryUpdate(data=memory, update=update, flag=flag, factor=update_factor)
    output2 = mx.symbol.MemoryUpdate(data=output, update=update, flag=flag, factor=update_factor)
    output2 = mx.symbol.BlockGrad(data=output2)
    data_shapes = {'memory': (5, 3, 2, 2), 'update': (1, 3, 2, 2), 'flag': (5, ),
                   'update_factor':(1,)}
    net = Base(sym=output2, data_shapes=data_shapes)

    memory_npy = numpy.zeros((5, 3, 2, 2), dtype=numpy.float32)
    update_npy = numpy.zeros((1, 3, 2, 2), dtype=numpy.float32)
    flag_npy = numpy.zeros((5,), dtype=numpy.float32)
    update_factor_npy = numpy.array([0.8,])
    for i in range(5):
        memory_npy[i, :, :, :] = 2*i + 1
    flag_npy[1] = 1
    output_npy = net.forward(data_shapes=data_shapes, memory=memory_npy, update=update_npy, flag=flag_npy,
                update_factor=update_factor_npy)[0].asnumpy()
    net.backward(data_shapes=data_shapes, memory=memory_npy, update=update_npy, flag=flag_npy,
                update_factor=update_factor_npy)
    print memory_npy
    print output_npy

def sum_channel_test():
    data = mx.symbol.Variable('data')
    summed_data = mx.symbol.SumChannel(data=data, dim=3)
    summed_data = mx.symbol.BlockGrad(data=summed_data)
    data_shapes = {'data': (10, 9, 8, 7)}
    net = Base(sym=summed_data, data_shapes=data_shapes)
    data_npy = numpy.ones((10,9,8,7))
    output_npy = net.forward(data_shapes=data_shapes, data=data_npy)[0].asnumpy()
    net.backward(data_shapes=data_shapes, data=data_npy)
    print output_npy
    print output_npy.shape

def complex_hadamard_test():
    ldata = mx.symbol.Variable('ldata')
    rdata = mx.symbol.Variable('rdata')
    product = mx.symbol.ComplexHadamard(ldata=ldata, rdata=rdata)
    product = mx.symbol.BlockGrad(data=product)
    data_shapes = {'ldata': (1, 1, 4, 4), 'rdata': (1, 1, 4, 4)}
    net = Base(sym=product, data_shapes=data_shapes)
    ldata_npy = numpy.ones((1, 1, 4, 4))
    rdata_npy = numpy.ones((1, 1, 4, 4))
    ldata_npy[0,0,0,0] = 2
    rdata_npy[0,0,1,0] = -1
    output_npy = net.forward(data_shapes=data_shapes, ldata=ldata_npy, rdata=rdata_npy)[0].asnumpy()
    net.backward(data_shapes=data_shapes, ldata=ldata_npy, rdata=rdata_npy)
    print output_npy
    print output_npy.shape

def complex_conjugate():
    data = mx.symbol.Variable('data')
    conjugate = mx.symbol.Conjugate(data=data)
    conjugate = mx.symbol.BlockGrad(data=conjugate)
    data_shapes = {'data': (1, 1, 4, 4)}
    net = Base(sym=conjugate, data_shapes=data_shapes)
    data_npy = numpy.ones((1, 1, 4, 4))
    output_npy = net.forward(data_shapes=data_shapes, data=data_npy)[0].asnumpy()
    net.backward(data_shapes=data_shapes, data=data_npy)
    print output_npy
    print output_npy.shape

def broadcast_channel():
    data = mx.symbol.Variable('data')
    broadcast = mx.symbol.BroadcastChannel(data=data, dim=0, size=10)
    broadcast = mx.symbol.BlockGrad(data=broadcast)
    data_shapes = {'data': (1, 1, 4, 4)}
    net = Base(sym=broadcast, data_shapes=data_shapes)
    data_npy = numpy.random.rand(1, 1, 4, 4)
    output_npy = net.forward(data_shapes=data_shapes, data=data_npy)[0].asnumpy()
    net.backward(data_shapes=data_shapes, data=data_npy)
    print output_npy
    print output_npy.shape

broadcast_channel()
complex_conjugate()
complex_hadamard_test()
sum_channel_test()
memory_update_test()
memory_choose_test()
memory_stat_read_write()
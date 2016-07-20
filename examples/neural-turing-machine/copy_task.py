import mxnet as mx
import mxnet.ndarray as nd
from mxnet.ops.recurrent import LSTM
import numpy
import matplotlib.pyplot as plt
from ntm import NTM
from arena import Base
from arena.utils import *

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

def gen_data(batch_size, data_dim, min_length, max_length):
    seqlen = numpy.random.randint(min_length, max_length + 1)
    data_in = numpy.random.randint(0, 2, size=(2 * seqlen + 2,  batch_size, data_dim), dtype=numpy.int)
    data_out = numpy.empty((seqlen, batch_size, data_dim), dtype=numpy.int)
    data_in[0, :, :] = 0
    data_in[0, :, 1] = 1
    data_in[seqlen + 1, :, :] = 0
    data_in[seqlen + 1, :, 2] = 1
    data_in[(seqlen + 2):, :, 2] = 0
    data_out[:] = data_in[1:(seqlen + 1), :, :]
    return seqlen, data_in, data_out

batch_size = 1
data_dim = 10
max_iter = 500
min_length = 10
max_length = 10
num_reads = 1
num_writes = 1
memory_size = 128
memory_state_dim = 20
control_state_dim = 100

def sym_gen(seqlen):
    print seqlen
    data_seqlen = 2*seqlen + 2
    data = mx.sym.Variable('data')
    target = mx.sym.Variable('target')
    data = mx.sym.SliceChannel(data, num_outputs=data_seqlen, axis=0, squeeze_axis=True) # (batch_size, data_dim) * seqlen
    # Initialize Memory
    init_memory = mx.sym.Variable('init_memory')
    init_read_focus = [mx.sym.Variable('NTM->read_head%d:init_focus' %i) for i in range(num_reads)]
    init_write_focus = [mx.sym.Variable('NTM->write_head%d:init_focus' % i) for i in range(num_writes)]
    # Initialize Control Network
    controller = LSTM(num_hidden=control_state_dim, name="controller")
    mem = NTM(num_reads=num_reads, num_writes=num_writes, memory_size=memory_size,
              memory_state_dim=memory_state_dim, control_state_dim=control_state_dim,
              init_memory=init_memory, init_read_focus=init_read_focus,
              init_write_focus=init_write_focus, name="NTM")
    controller_h = controller.init_h[0]
    controller_c = controller.init_c[0]
    read_content_l, read_focus_l = mem.read(controller_h)
    erase_signal_l, add_signal_l, write_focus_l = mem.write(controller_h)
    controller_states = []
    # Processing the start_symbol + input_sequence + end_symbol
    for i in range(data_seqlen):
        controller_h, controller_c =\
            controller.step(data=mx.sym.Concat(data[i], *read_content_l,
                                               num_args=1 + len(read_focus_l)),
                            prev_h=controller_h,
                            prev_c=controller_c,
                            seq_length=1)
        controller_h = controller_h[0]
        controller_c = controller_c[0]
        read_content_l, read_focus_l = mem.read(controller_h)
        erase_signal_l, add_signal_l, write_focus_l = mem.write(controller_h)
        controller_states.append(controller_h)
    print len(controller_states[(seqlen + 2):])
    output_state = mx.sym.Concat(*controller_states[seqlen+2:], num_args=seqlen, dim=0)
    prediction = mx.sym.FullyConnected(data=output_state, num_hidden=data_dim, name="fc")
    prediction = mx.sym.Activation(prediction, act_type='sigmoid', name="pred")
    out_sym = mx.sym.LogisticRegressionOutput(data=mx.sym.Reshape(prediction, shape=(-1, 1)),
                                              label=mx.sym.Reshape(target, shape=(-1, 1)))
    return mx.sym.Group([out_sym, mx.sym.BlockGrad(prediction)])

def binary_entropy_loss(prediction, target):
    log_pred = numpy.log(prediction + 1E-9)
    return - (numpy.log(prediction + 1E-9) * target +
              numpy.log(1 - prediction + 1E-9) * (1 - target)).sum()

def update_line(hl, fig, ax, new_x, new_y):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_x))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_y))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
plt.ion()
fig, ax = plt.subplots()
lines, = ax.plot([], [])

ax.set_autoscaley_on(True)
max_input_seq_len = max_length*2 + 2
max_output_seq_len = max_length
net = Base(data_shapes={'data': (max_input_seq_len, batch_size, data_dim),
                        'target': (max_output_seq_len, batch_size, data_dim),
                        'init_memory': (batch_size, memory_size, memory_state_dim),
                        'NTM->read_head0:init_focus': (batch_size, memory_size),
                        'NTM->write_head0:init_focus': (batch_size, memory_size),
                        'controller->layer0:init_h': (batch_size, control_state_dim),
                        'controller->layer0:init_c': (batch_size, control_state_dim)},
           sym_gen=sym_gen,
           default_bucket_kwargs={'seqlen': max_length})
net.print_stat()
init_memory_npy = numpy.zeros((batch_size, memory_size, memory_state_dim), dtype=numpy.float32) + 0.1#numpy.tanh(numpy.random.normal(size=(batch_size, memory_size, memory_state_dim)))
# init_read_focus_npy = npy_softmax(
#                       numpy.broadcast_to(numpy.arange(memory_size, 0, -1), (batch_size, memory_size)),
#                       axis=1)
# init_write_focus_npy = npy_softmax(
#                        numpy.broadcast_to(numpy.arange(memory_size, 0, -1), (batch_size, memory_size)),
#                        axis=1)
init_read_focus_npy = npy_softmax(numpy.ones((batch_size, memory_size), dtype=numpy.float32), axis=1)
init_write_focus_npy = npy_softmax(numpy.ones((batch_size, memory_size), dtype=numpy.float32), axis=1)
init_h_npy = numpy.zeros((batch_size, control_state_dim), dtype=numpy.float32) + 0.0001#numpy.tanh(numpy.random.normal(size=(batch_size, control_state_dim)))
init_c_npy = numpy.zeros((batch_size, control_state_dim), dtype=numpy.float32) + 0.0001#numpy.tanh(numpy.random.normal(size=(batch_size, control_state_dim)))
optimizer = mx.optimizer.create(name='SGD', learning_rate=1E-3, momentum=0.9)
updater = mx.optimizer.get_updater(optimizer)
for i in range(max_iter):
    seqlen, data_in, data_out = gen_data(batch_size=batch_size, data_dim=data_dim,
                                         min_length=min_length, max_length=max_length)
    print data_in.shape
    print seqlen
    print data_out.shape
    outputs =\
        net.forward(is_train=True, bucket_kwargs={'seqlen': seqlen},
                    **{'data': data_in,
                       'target': data_out,
                       'init_memory': init_memory_npy,
                       'NTM->read_head0:init_focus': init_read_focus_npy,
                       'NTM->write_head0:init_focus': init_write_focus_npy,
                       'controller->layer0:init_h': init_h_npy,
                       'controller->layer0:init_c': init_c_npy})
    net.backward()
    norm_clipping(net.params_grad, 1)
    net.update(updater=updater)
    avg_loss = binary_entropy_loss(outputs[1].asnumpy(), data_out)/seqlen
    print avg_loss
    update_line(lines, fig, ax, i, avg_loss)

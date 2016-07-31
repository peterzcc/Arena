import mxnet as mx
import mxnet.ndarray as nd
import numpy
import matplotlib.pyplot as plt
from ntm import NTM
from arena import Base
from arena.helpers.visualization import *
from arena.ops import LSTM
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
    data_in = numpy.random.randint(0, 2, size=(2 * seqlen + 2,  batch_size, data_dim), dtype=numpy.int).astype(numpy.float32)
    data_out = numpy.empty((seqlen, batch_size, data_dim), dtype=numpy.int).astype(numpy.float32)
    data_in[0, :, :] = 0
    data_in[0, :, 0] = 0.5
    data_in[seqlen + 1, :, :] = 0
    data_in[seqlen + 1, :, 1] = 0.5
    data_in[(seqlen + 2):, :, :] = 0
    data_out[:] = data_in[1:(seqlen + 1), :, :]
    return seqlen, data_in, data_out

def gen_data_same(batch_size, data_dim, min_length, max_length):
    seqlen = numpy.random.randint(min_length, max_length + 1)
    data_in = numpy.ones((2 * seqlen + 2,  batch_size, data_dim), dtype=numpy.float32)
    data_out = numpy.empty((seqlen, batch_size, data_dim), dtype=numpy.int).astype(numpy.float32)
    data_in[0, :, :] = 0
    data_in[0, :, 0] = 0.5
    data_in[seqlen + 1, :, :] = 0
    data_in[seqlen + 1, :, 1] = 0.5
    data_in[(seqlen + 2):, :, :] = 0
    data_out[:] = data_in[1:(seqlen + 1), :, :]
    return seqlen, data_in, data_out

batch_size = 16
data_dim = 10
max_iter = 2500
min_length = 1
max_length = 20
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

    init_read_focus = [mx.sym.Variable('NTM->read_head%d:init_focus' %i) for i in range(num_reads)] # TODO
    init_write_focus = [mx.sym.Variable('NTM->write_head%d:init_focus' % i) for i in range(num_writes)] # TODO
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
    all_read_focus_l = []
    all_write_focus_l = []
    all_read_content_l = []
    all_erase_signal_l = []
    all_add_signal_l = []
    # Processing the start_symbol + input_sequence + end_symbol
    for i in range(data_seqlen):
        controller_h, controller_c =\
            controller.step(data=mx.sym.Concat(data[i], *read_content_l,
                                               num_args=1 + len(read_content_l)),
                            prev_h=controller_h,
                            prev_c=controller_c,
                            seq_length=1)
        controller_h = controller_h[0]
        controller_c = controller_c[0]
        read_content_l, read_focus_l = mem.read(controller_h)
        erase_signal_l, add_signal_l, write_focus_l = mem.write(controller_h)
        all_read_focus_l.append(read_focus_l[0])
        all_write_focus_l.append(write_focus_l[0])
        all_read_content_l.append(read_content_l[0])
        all_erase_signal_l.append(erase_signal_l[0])
        all_add_signal_l.append(add_signal_l[0])
        controller_states.append(controller_h)
    output_state = mx.sym.Concat(*controller_states[(seqlen+2):], num_args=seqlen, dim=0)
    prediction = mx.sym.FullyConnected(data=output_state, num_hidden=data_dim, name="fc")
    target = mx.sym.Reshape(target, shape=(-1, 1))
    out_sym = mx.sym.LogisticRegressionOutput(data=mx.sym.Reshape(prediction, shape=(-1, 1)),
                                              label=target)
    return mx.sym.Group([out_sym,
                         mx.sym.BlockGrad(mx.sym.Reshape(
                             mx.sym.Concat(*controller_states, dim=0,
                                           num_args=len(controller_states)),
                             shape=(data_seqlen, -1, control_state_dim))),
                         mx.sym.BlockGrad(mx.sym.Reshape(
                             mx.sym.Concat(*all_read_focus_l, dim=0,
                                           num_args=len(all_read_focus_l)),
                             shape=(data_seqlen, -1, memory_size))),
                         mx.sym.BlockGrad(mx.sym.Reshape(
                             mx.sym.Concat(*all_write_focus_l, dim=0,
                                           num_args=len(all_write_focus_l)),
                             shape=(data_seqlen, -1, memory_size))),
                         mx.sym.BlockGrad(mx.sym.Reshape(
                             mx.sym.Concat(*all_read_content_l, dim=0,
                                           num_args=len(all_read_content_l)),
                             shape=(data_seqlen, -1, memory_state_dim))),
                         mx.sym.BlockGrad(mx.sym.Reshape(
                             mx.sym.Concat(*all_erase_signal_l, dim=0,
                                           num_args=len(all_erase_signal_l)),
                             shape=(data_seqlen, -1, memory_state_dim))),
                         mx.sym.BlockGrad(mx.sym.Reshape(
                             mx.sym.Concat(*all_add_signal_l, dim=0,
                                           num_args=len(all_add_signal_l)),
                             shape=(data_seqlen, -1, memory_state_dim)))
                         ])

vis = PLTVisualizer()

max_input_seq_len = max_length*2 + 2
max_output_seq_len = max_length
sym = sym_gen(max_length)
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
init_memory_npy = numpy.tanh(numpy.random.normal(size=(batch_size, memory_size, memory_state_dim)))
# init_memory_npy = numpy.zeros((batch_size, memory_size, memory_state_dim), dtype=numpy.float32) + 0.1
init_read_focus_npy = npy_softmax(
                      numpy.broadcast_to(numpy.arange(memory_size, 0, -1), (batch_size, memory_size)),
                      axis=1)
init_write_focus_npy = npy_softmax(
                       numpy.broadcast_to(numpy.arange(memory_size, 0, -1), (batch_size, memory_size)),
                       axis=1)
# init_read_focus_npy = npy_softmax(numpy.ones((batch_size, memory_size), dtype=numpy.float32), axis=1)
# init_write_focus_npy = npy_softmax(numpy.ones((batch_size, memory_size), dtype=numpy.float32), axis=1)

# init_read_focus_npy = npy_softmax(numpy.random.normal(size=(batch_size, memory_size)), axis=1)
# init_write_focus_npy = npy_softmax(numpy.random.normal(size=(batch_size, memory_size)), axis=1)

init_h_npy = numpy.zeros((batch_size, control_state_dim), dtype=numpy.float32) + 0.0001#numpy.tanh(numpy.random.normal(size=(batch_size, control_state_dim)))
init_c_npy = numpy.zeros((batch_size, control_state_dim), dtype=numpy.float32) + 0.0001#numpy.tanh(numpy.random.normal(size=(batch_size, control_state_dim)))
optimizer = mx.optimizer.create(name='RMSProp', learning_rate=1E-4, rescale_grad=1.0/batch_size)
updater = mx.optimizer.get_updater(optimizer)

for i in range(max_iter):
    seqlen, data_in, data_out = gen_data(batch_size=batch_size, data_dim=data_dim,
                                         min_length=min_length, max_length=max_length)
    print data_in.shape
    print seqlen
    print data_out.shape
    outputs =\
        net.forward(is_train=True,
                    bucket_kwargs={'seqlen': seqlen},
                    **{'data': data_in,
                       'target': data_out,
                       'init_memory': init_memory_npy,
                       'NTM->read_head0:init_focus': init_read_focus_npy,
                       'NTM->write_head0:init_focus': init_write_focus_npy,
                       'controller->layer0:init_h': init_h_npy,
                       'controller->layer0:init_c': init_c_npy})
    net.backward()
    norm_clipping(net.params_grad, 10)
    net.update(updater=updater)
    # for k, v in net.params.items():
    #     print k, nd.norm(v).asnumpy()
    for k, v in net.params_grad.items():
        print k, nd.norm(v).asnumpy()
    pred = outputs[0].reshape((seqlen, batch_size, data_dim)).asnumpy()
    state_over_time = outputs[1].asnumpy()
    read_weight_over_time = outputs[2].asnumpy()
    write_weight_over_time = outputs[3].asnumpy()
    read_content_over_time = outputs[4].asnumpy()
    erase_signal_over_time = outputs[5].asnumpy()
    add_signal_over_time = outputs[6].asnumpy()
    cv2_visualize(data=pred[:, 0, :].T, win_name="prediction")
    cv2_visualize(data=data_out[:, 0, :].T, win_name="target")
    cv2_visualize(data=state_over_time[:, 0, :].T, win_name="state")
    cv2_visualize(data=read_weight_over_time[:, 0, :].T, win_name="read_weight")
    cv2_visualize(data=write_weight_over_time[:, 0, :].T, win_name="write_weight")
    cv2_visualize(data=(read_content_over_time[:, 0, :].T + 1) / 2,
                          win_name="read_content")
    cv2_visualize(data=erase_signal_over_time[:, 0, :].T,
                          win_name="erase_signal")
    cv2_visualize(data=(add_signal_over_time[:, 0, :].T + 1) / 2,
                          win_name="add_signal")

    #visualize_weights(state_over_time[:, 0, :], win_name="state")
    #visualize_weights(read_weight_over_time[:, 0, :], win_name="read_weight")
    #visualize_weights(write_weight_over_time[:, 0, :], win_name="write_weight")
    avg_loss = npy_binary_entropy(pred, data_out)/seqlen/batch_size
    print avg_loss
    vis.update(i, avg_loss)

test_seq_len_l = [120, 129]
for j in range(2):
    for i in range(3):
        seqlen, data_in, data_out = gen_data(batch_size=batch_size, data_dim=data_dim,
                                             min_length=test_seq_len_l[j],
                                             max_length=test_seq_len_l[j])
        print data_in.shape
        print seqlen
        print data_out.shape
        outputs =\
            net.forward(is_train=False,
                        bucket_kwargs={'seqlen': seqlen},
                        **{'data': data_in,
                           'target': data_out,
                           'init_memory': init_memory_npy,
                           'NTM->read_head0:init_focus': init_read_focus_npy,
                           'NTM->write_head0:init_focus': init_write_focus_npy,
                           'controller->layer0:init_h': init_h_npy,
                           'controller->layer0:init_c': init_c_npy})
        pred = outputs[0].reshape((seqlen, batch_size, data_dim)).asnumpy()
        state_over_time = outputs[1].asnumpy()
        read_weight_over_time = outputs[2].asnumpy()
        write_weight_over_time = outputs[3].asnumpy()
        read_content_over_time = outputs[4].asnumpy()
        erase_signal_over_time = outputs[5].asnumpy()
        add_signal_over_time = outputs[6].asnumpy()
        cv2_visualize(data=pred[:, 0, :].T, win_name="prediction")
        cv2_visualize(data=data_out[:, 0, :].T, win_name="target")
        cv2_visualize(data=state_over_time[:, 0, :].T, win_name="state")
        cv2_visualize(data=read_weight_over_time[:, 0, :].T,
                              win_name="read_weight")
        cv2_visualize(data=write_weight_over_time[:, 0, :].T,
                              win_name="write_weight")
        cv2_visualize(data=(read_content_over_time[:, 0, :].T + 1) / 2,
                              win_name="read_content")
        cv2_visualize(data=erase_signal_over_time[:, 0, :].T, win_name="erase_signal")
        cv2_visualize(data=(add_signal_over_time[:, 0, :].T + 1) / 2, win_name="add_signal")

        cv2_save(data=pred[:, 0, :].T,
                           path="./prediction_seqlen%d_%d.jpg" %(seqlen, i))
        cv2_save(data=data_out[:, 0, :].T,
                           path="./target_seqlen%d_%d.jpg" %(seqlen, i))
        cv2_save(data=read_weight_over_time[:, 0, :].T,
                           path="./read_weight_seqlen%d_%d.jpg")
        cv2_save(data=write_weight_over_time[:, 0, :].T,
                           path="./write_weight_seqlen%d_%d.jpg")
        cv2_save(data=(read_content_over_time[:, 0, :].T + 1) / 2,
                           path="./read_content_seqlen%d_%d.jpg")
        avg_loss = npy_binary_entropy(pred, data_out)/seqlen/batch_size
        print(avg_loss)
        ch = raw_input()

test_seq_len_l = [120, 129]
for j in range(2):
    for i in range(3):
        seqlen, data_in, data_out = gen_data_same(batch_size=batch_size, data_dim=data_dim,
                                             min_length=test_seq_len_l[j],
                                             max_length=test_seq_len_l[j])
        print data_in.shape
        print seqlen
        print data_out.shape
        outputs =\
            net.forward(is_train=False,
                        bucket_kwargs={'seqlen': seqlen},
                        **{'data': data_in,
                           'target': data_out,
                           'init_memory': init_memory_npy,
                           'NTM->read_head0:init_focus': init_read_focus_npy,
                           'NTM->write_head0:init_focus': init_write_focus_npy,
                           'controller->layer0:init_h': init_h_npy,
                           'controller->layer0:init_c': init_c_npy})
        pred = outputs[0].reshape((seqlen, batch_size, data_dim)).asnumpy()
        state_over_time = outputs[1].asnumpy()
        read_weight_over_time = outputs[2].asnumpy()
        write_weight_over_time = outputs[3].asnumpy()
        read_content_over_time = outputs[4].asnumpy()
        erase_signal_over_time = outputs[5].asnumpy()
        add_signal_over_time = outputs[6].asnumpy()
        cv2_visualize(data=pred[:, 0, :].T, win_name="prediction")
        cv2_visualize(data=data_out[:, 0, :].T, win_name="target")
        cv2_visualize(data=state_over_time[:, 0, :].T, win_name="state")
        cv2_visualize(data=read_weight_over_time[:, 0, :].T,
                              win_name="read_weight")
        cv2_visualize(data=write_weight_over_time[:, 0, :].T,
                              win_name="write_weight")
        cv2_visualize(data=(read_content_over_time[:, 0, :].T + 1) / 2,
                              win_name="read_content")
        cv2_visualize(data=erase_signal_over_time[:, 0, :].T, win_name="erase_signal")
        cv2_visualize(data=(add_signal_over_time[:, 0, :].T + 1) / 2, win_name="add_signal")

        cv2_save(data=pred[:, 0, :].T,
                           path="./same_prediction_seqlen%d_%d.jpg" %(seqlen, i))
        cv2_save(data=data_out[:, 0, :].T,
                           path="./same_target_seqlen%d_%d.jpg" %(seqlen, i))
        cv2_save(data=read_weight_over_time[:, 0, :].T,
                           path="./same_read_weight_seqlen%d_%d.jpg")
        cv2_save(data=write_weight_over_time[:, 0, :].T,
                           path="./same_write_weight_seqlen%d_%d.jpg")
        cv2_save(data=(read_content_over_time[:, 0, :].T + 1) / 2,
                           path="./same_read_content_seqlen%d_%d.jpg")
        avg_loss = npy_binary_entropy(pred, data_out)/seqlen/batch_size
        print(avg_loss)
        ch = raw_input()

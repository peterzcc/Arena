# coding: utf-8
from __future__ import absolute_import, division, print_function
from builtins import input

from arena import Base
from arena.helpers.visualization import *
from arena.ops import *
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
max_iter = 5000
min_length = 1
max_length = 20
num_reads = 1
num_writes = 1
memory_size = 128
memory_state_dim = 20
control_state_dim = 100

def sym_gen(seqlen):
    print(seqlen)
    data_seqlen = 2*seqlen + 2
    data = mx.sym.Variable('data')
    target = mx.sym.Variable('target')
    data = mx.sym.SliceChannel(data, num_outputs=data_seqlen, axis=0, squeeze_axis=False) # (batch_size, data_dim) * seqlen
    # Initialize Memory
    init_memory = mx.sym.Variable('init_memory')
<<<<<<< HEAD

    init_read_focus = mx.sym.Variable('NTM->read_head:init_focus')
    init_write_focus = mx.sym.Variable('NTM->write_head:init_focus')
=======
    init_read_content = mx.sym.Variable('init_read_content')
    init_read_focus = mx.sym.Variable('NTM->read_head:init_focus')
    init_write_focus = mx.sym.Variable('NTM->write_head:init_focus')
    init_h = [mx.sym.Variable('controller->layer0:init_h')]
    init_c = [mx.sym.Variable('controller->layer0:init_c')]

    init_memory = mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(init_memory, act_type="tanh"),
                                                         axis=0),
                                      shape=(batch_size, memory_size, memory_state_dim))
    # init_memory = mx.sym.Custom(init_memory, op_type="Identity")
    init_read_content = mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(init_read_content,
                                                                                 act_type="tanh"),
                                                               axis=0),
                                            shape=(batch_size, num_reads, memory_state_dim))
    init_read_focus = mx.sym.SoftmaxActivation(init_read_focus)
    init_read_focus = mx.sym.broadcast_to(mx.sym.expand_dims(init_read_focus, axis=0),
                                          shape=(batch_size, num_reads, memory_size))
    init_write_focus = mx.sym.SoftmaxActivation(init_write_focus)
    init_write_focus = mx.sym.broadcast_to(mx.sym.expand_dims(init_write_focus, axis=0),
                                           shape=(batch_size, num_writes, memory_size))
    init_h = [mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(ele, act_type="tanh"),
                                                     axis=0),
                                  shape=(batch_size, control_state_dim))
              for ele in init_h]
    init_c = [mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(ele, act_type="tanh"),
                                                     axis=0),
                                  shape=(batch_size, control_state_dim))
              for ele in init_c]
>>>>>>> arena

    # Initialize Control Network
    controller = RNN(num_hidden=[control_state_dim], data_dim=data_dim + num_reads * memory_state_dim,
                     typ="lstm",
                     init_h=init_h,
                     init_c=init_c,
                     name="controller")
    mem = NTM(num_reads=num_reads, num_writes=num_writes, memory_size=memory_size,
              memory_state_dim=memory_state_dim, control_state_dim=control_state_dim,
              init_memory=init_memory, init_read_focus=init_read_focus,
              init_write_focus=init_write_focus, name="NTM")
    controller_h = controller.init_h[0]
    controller_c = controller.init_c[0]
    # read_content, read_focus = mem.read(controller_h)
    # erase_signal, add_signal, write_focus = mem.write(controller_h)
    read_content = init_read_content
    controller_states = []
    all_read_focus_l = []
    all_write_focus_l = []
    all_read_content_l = []
    all_erase_signal_l = []
    all_add_signal_l = []
    # Processing the start_symbol + input_sequence + end_symbol
    for i in range(data_seqlen):
        controller_h, controller_c =\
            controller.step(data=mx.sym.Concat(data[i],
                              mx.sym.Reshape(read_content, shape=(1, -1, num_reads * memory_state_dim)),
                              num_args=2, dim=2),
                            prev_h=controller_h,
                            prev_c=controller_c,
                            seq_len=1,
                            ret_typ="state")
        controller_h = controller_h[0]
        controller_c = controller_c[0]
        read_content, read_focus = mem.read(controller_h)
        erase_signal, add_signal, write_focus = mem.write(controller_h)
        all_read_focus_l.append(read_focus)
        all_write_focus_l.append(write_focus)
        all_read_content_l.append(read_content)
        all_erase_signal_l.append(erase_signal)
        all_add_signal_l.append(add_signal)
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
                             shape=(data_seqlen, -1, num_reads, memory_size))),
                         mx.sym.BlockGrad(mx.sym.Reshape(
                             mx.sym.Concat(*all_write_focus_l, dim=0,
                                           num_args=len(all_write_focus_l)),
                             shape=(data_seqlen, -1, num_writes, memory_size))),
                         mx.sym.BlockGrad(mx.sym.Reshape(
                             mx.sym.Concat(*all_read_content_l, dim=0,
                                           num_args=len(all_read_content_l)),
                             shape=(data_seqlen, -1, num_reads, memory_state_dim))),
                         mx.sym.BlockGrad(mx.sym.Reshape(
                             mx.sym.Concat(*all_erase_signal_l, dim=0,
                                           num_args=len(all_erase_signal_l)),
                             shape=(data_seqlen, -1, num_writes, memory_state_dim))),
                         mx.sym.BlockGrad(mx.sym.Reshape(
                             mx.sym.Concat(*all_add_signal_l, dim=0,
                                           num_args=len(all_add_signal_l)),
                             shape=(data_seqlen, -1, num_writes, memory_state_dim)))
                         ])
class NTMInitializer(mx.init.Xavier):
    def _init_default(self, name, arr):
        if name == "init_memory":
            arr[:] = numpy.random.normal(loc=0.1, size=arr.shape)
        elif name == "init_read_content":
            arr[:] = numpy.random.normal(loc=0.1, size=arr.shape)
        elif ("read_head" in name or "write_head" in name) and "init_focus" in name:
            arr[:] = numpy.broadcast_to(numpy.arange(arr.shape[-1], 0, -1), arr.shape)
        elif "init_h" in name or "init_c" in name:
            arr[:] = numpy.random.normal(loc=0.1, size=arr.shape)
        else:
            assert False
vis = PLTVisualizer()



max_input_seq_len = max_length*2 + 2
max_output_seq_len = max_length
sym = sym_gen(max_length)
net = Base(data_shapes={'data': (max_input_seq_len, batch_size, data_dim),
                        'target': (max_output_seq_len, batch_size, data_dim),
                        'init_memory': (memory_size, memory_state_dim),
                        'init_read_content': (num_reads, memory_state_dim),
                        'NTM->read_head:init_focus': (num_reads, memory_size),
                        'NTM->write_head:init_focus': (num_writes, memory_size),
                        'controller->layer0:init_h': (control_state_dim,),
                        'controller->layer0:init_c': (control_state_dim,)},
           sym_gen=sym_gen,
           learn_init_keys=['init_memory',
                            'init_read_content',
                            'NTM->read_head:init_focus',
                            'NTM->write_head:init_focus',
                            'controller->layer0:init_h',
                            'controller->layer0:init_c'],
           default_bucket_kwargs={'seqlen': max_length},
           initializer=NTMInitializer(factor_type="in", rnd_type="gaussian", magnitude=2),
           ctx=mx.gpu())
net.print_stat()
###init_memory_npy = numpy.tanh(numpy.random.normal(size=(batch_size, memory_size, memory_state_dim)))
# init_memory_npy = numpy.zeros((batch_size, memory_size, memory_state_dim), dtype=numpy.float32) + 0.1
# init_read_focus_npy = numpy.random.randint(0, memory_size, size=(batch_size, num_reads))
# init_read_focus_npy = npy_softmax(npy_onehot(init_read_focus_npy, num=memory_size), axis=2)
###init_read_focus_npy = npy_softmax(
###                      numpy.broadcast_to(numpy.arange(memory_size, 0, -1), (batch_size, num_reads, memory_size)),
###                      axis=2)
# init_write_focus_npy = numpy.random.randint(0, memory_size, size=(batch_size, num_writes))
# init_write_focus_npy = npy_softmax(npy_onehot(init_write_focus_npy, num=memory_size), axis=2)
###init_write_focus_npy = npy_softmax(
###                       numpy.broadcast_to(numpy.arange(memory_size, 0, -1), (batch_size, num_writes, memory_size)),
###                       axis=2)

# init_read_focus_npy = npy_softmax(numpy.ones((batch_size, memory_size), dtype=numpy.float32), axis=1)
# init_write_focus_npy = npy_softmax(numpy.ones((batch_size, memory_size), dtype=numpy.float32), axis=1)

# init_read_focus_npy = npy_softmax(numpy.random.normal(size=(batch_size, memory_size)), axis=1)
# init_write_focus_npy = npy_softmax(numpy.random.normal(size=(batch_size, memory_size)), axis=1)

###init_h_npy = numpy.zeros((batch_size, control_state_dim), dtype=numpy.float32) + 0.0001#numpy.tanh(numpy.random.normal(size=(batch_size, control_state_dim)))
###init_c_npy = numpy.zeros((batch_size, control_state_dim), dtype=numpy.float32) + 0.0001#numpy.tanh(numpy.random.normal(size=(batch_size, control_state_dim)))
optimizer = mx.optimizer.create(name='RMSProp', learning_rate=1E-4, rescale_grad=1.0/batch_size)
updater = mx.optimizer.get_updater(optimizer)

for i in range(max_iter):
    seqlen, data_in, data_out = gen_data(batch_size=batch_size, data_dim=data_dim,
                                         min_length=min_length, max_length=max_length)
    print(data_in.shape)
    print(seqlen)
    print(data_out.shape)
    outputs =\
        net.forward(is_train=True,
                    bucket_kwargs={'seqlen': seqlen},
                    **{'data': data_in,
                       'target': data_out})
    net.backward()
    norm_clipping(net.params_grad, 10)
    net.update(updater=updater)
    # for k, v in net.params.items():
    #     print k, nd.norm(v).asnumpy()
    for k, v in net.params_grad.items():
        print(k, nd.norm(v).asnumpy())
    pred = outputs[0].reshape((seqlen, batch_size, data_dim)).asnumpy()
    state_over_time = outputs[1].asnumpy()
    read_weight_over_time = outputs[2].asnumpy()
    write_weight_over_time = outputs[3].asnumpy()
    read_content_over_time = outputs[4].asnumpy()
    erase_signal_over_time = outputs[5].asnumpy()
    add_signal_over_time = outputs[6].asnumpy()
    CV2Vis.display(data=pred[:, 0, :].T, win_name="prediction")
    CV2Vis.display(data=data_out[:, 0, :].T, win_name="target")
    CV2Vis.display(data=state_over_time[:, 0, :].T, win_name="state")
    for read_id in range(num_reads):
        CV2Vis.display(data=read_weight_over_time[:, 0, read_id, :].T,
                      win_name="read_weight%d" %read_id)
        CV2Vis.display(data=(read_content_over_time[:, 0, read_id, :].T + 1) / 2,
                      win_name="read_content%d" %read_id)
    for write_id in range(num_writes):
        CV2Vis.display(data=write_weight_over_time[:, 0, write_id, :].T,
                      win_name="write_weight%d" %write_id)
        CV2Vis.display(data=erase_signal_over_time[:, 0, write_id, :].T,
                      win_name="erase_signal%d" %write_id)
        CV2Vis.display(data=(add_signal_over_time[:, 0, write_id, :].T + 1) / 2,
                      win_name="add_signal%d" %write_id)
    avg_loss = npy_binary_entropy(pred, data_out)/seqlen/batch_size
    print(avg_loss)
    vis.update(i, avg_loss)

test_seq_len_l = [120, 129]
for j in range(2):
    for i in range(3):
        seqlen, data_in, data_out = gen_data(batch_size=batch_size, data_dim=data_dim,
                                             min_length=test_seq_len_l[j],
                                             max_length=test_seq_len_l[j])
        print(data_in.shape)
        print(seqlen)
        print(data_out.shape)
        outputs =\
            net.forward(is_train=False,
                        bucket_kwargs={'seqlen': seqlen},
                        **{'data': data_in,
                           'target': data_out})
        pred = outputs[0].reshape((seqlen, batch_size, data_dim)).asnumpy()
        state_over_time = outputs[1].asnumpy()
        read_weight_over_time = outputs[2].asnumpy()
        write_weight_over_time = outputs[3].asnumpy()
        read_content_over_time = outputs[4].asnumpy()
        erase_signal_over_time = outputs[5].asnumpy()
        add_signal_over_time = outputs[6].asnumpy()
        CV2Vis.display(data=pred[:, 0, :].T, win_name="prediction", save_image=True,
                      save_path="./prediction_seqlen%d_%d.jpg" %(seqlen, i))
        CV2Vis.display(data=data_out[:, 0, :].T, win_name="target", save_image=True,
                      save_path="./target_seqlen%d_%d.jpg" %(seqlen, i))
        CV2Vis.display(data=state_over_time[:, 0, :].T, win_name="state", save_image=True,
                      save_path="./state_seqlen%d_%d.jpg" %(seqlen, i))
        for read_id in range(num_reads):
            CV2Vis.display(data=read_weight_over_time[:, 0, read_id, :].T,
                          win_name="read_weight%d" % read_id,
                          save_image=True,
                          save_path="./read_weight_seqlen%d_%d.jpg" %(seqlen, i))
            CV2Vis.display(data=(read_content_over_time[:, 0, read_id, :].T + 1) / 2,
                          win_name="read_content%d" % read_id,
                          save_image=True,
                          save_path="./read_content_seqlen%d_%d.jpg" %(seqlen, i))
        for write_id in range(num_writes):
            CV2Vis.display(data=write_weight_over_time[:, 0, write_id, :].T,
                          win_name="write_weight%d" % write_id,
                          save_image=True,
                          save_path="./write_weight_seqlen%d_%d.jpg" %(seqlen, i))
            CV2Vis.display(data=erase_signal_over_time[:, 0, write_id, :].T,
                          win_name="erase_signal%d" % write_id,
                          save_image=True,
                          save_path="./erase_signal_seqlen%d_%d.jpg" %(seqlen, i))
            CV2Vis.display(data=(add_signal_over_time[:, 0, write_id, :].T + 1) / 2,
                          win_name="add_signal%d" % write_id,
                          save_image=True,
                          save_path="./add_signal_seqlen%d_%d.jpg" %(seqlen, i))
        avg_loss = npy_binary_entropy(pred, data_out)/seqlen/batch_size
        print(avg_loss)
        ch = input()

test_seq_len_l = [120, 129]
for j in range(2):
    for i in range(3):
        seqlen, data_in, data_out = gen_data_same(batch_size=batch_size, data_dim=data_dim,
                                             min_length=test_seq_len_l[j],
                                             max_length=test_seq_len_l[j])
        print(data_in.shape)
        print(seqlen)
        print(data_out.shape)
        outputs =\
            net.forward(is_train=False,
                        bucket_kwargs={'seqlen': seqlen},
                        **{'data': data_in,
                           'target': data_out})
        pred = outputs[0].reshape((seqlen, batch_size, data_dim)).asnumpy()
        state_over_time = outputs[1].asnumpy()
        read_weight_over_time = outputs[2].asnumpy()
        write_weight_over_time = outputs[3].asnumpy()
        read_content_over_time = outputs[4].asnumpy()
        erase_signal_over_time = outputs[5].asnumpy()
        add_signal_over_time = outputs[6].asnumpy()
        CV2Vis.display(data=pred[:, 0, :].T, win_name="prediction", save_image=True,
                       save_path="./same_prediction_seqlen%d_%d.jpg" %(seqlen, i))
        CV2Vis.display(data=data_out[:, 0, :].T, win_name="target", save_image=True,
                       save_path="./same_target_seqlen%d_%d.jpg" %(seqlen, i))
        CV2Vis.display(data=state_over_time[:, 0, :].T, win_name="state", save_image=True,
                       save_path="./same_state_seqlen%d_%d.jpg" %(seqlen, i))
        for read_id in range(num_reads):
            CV2Vis.display(data=read_weight_over_time[:, 0, read_id, :].T,
                           win_name="read_weight%d" % read_id,
                           save_image=True,
                           save_path="./same_read_weight_seqlen%d_%d.jpg" %(seqlen, i))
            CV2Vis.display(data=(read_content_over_time[:, 0, read_id, :].T + 1) / 2,
                          win_name="read_content%d" % read_id,
                          save_image=True,
                          save_path="./same_read_content_seqlen%d_%d.jpg" %(seqlen, i))
        for write_id in range(num_writes):
            CV2Vis.display(data=write_weight_over_time[:, 0, write_id, :].T,
                           win_name="write_weight%d" % write_id,
                           save_image=True,
                           save_path="./same_write_weight_seqlen%d_%d.jpg" %(seqlen, i))
            CV2Vis.display(data=erase_signal_over_time[:, 0, write_id, :].T,
                           win_name="erase_signal%d" % write_id,
                           save_image=True,
                           save_path="./same_erase_signal_seqlen%d_%d.jpg" %(seqlen, i))
            CV2Vis.display(data=(add_signal_over_time[:, 0, write_id, :].T + 1) / 2,
                           win_name="add_signal%d" % write_id,
                           save_image=True,
                           save_path="./same_add_signal_seqlen%d_%d.jpg" %(seqlen, i))
        avg_loss = npy_binary_entropy(pred, data_out)/seqlen/batch_size
        print(avg_loss)
        ch = input()

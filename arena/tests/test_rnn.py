import mxnet as mx
import mxnet.ndarray as nd
import numpy
import time
from mxnet.test_utils import check_numeric_gradient, reldiff
from arena.ops.recurrent import RNN, get_rnn_param_shapes

def step_vanilla_rnn(num_hidden, data, prev_h, act_f,
                     i2h_weight, i2h_bias, h2h_weight, h2h_bias, name):
    i2h = mx.sym.FullyConnected(data=data,
                                weight=i2h_weight,
                                bias=i2h_bias,
                                num_hidden=num_hidden,
                                name=name + ":i2h")
    h2h = mx.sym.FullyConnected(data=prev_h,
                                weight=h2h_weight,
                                bias=h2h_bias,
                                num_hidden=num_hidden,
                                name=name + ":h2h")
    new_h = act_f(i2h + h2h)
    return new_h

def step_relu_rnn(num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight, h2h_bias, name):
    return step_vanilla_rnn(num_hidden=num_hidden, data=data, prev_h=prev_h,
                            act_f=lambda x: mx.sym.Activation(x, act_type="relu"),
                            i2h_weight=i2h_weight, i2h_bias=i2h_bias,
                            h2h_weight=h2h_weight, h2h_bias=h2h_bias, name=name)

def step_tanh_rnn(num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight, h2h_bias, name):
    return step_vanilla_rnn(num_hidden=num_hidden, data=data, prev_h=prev_h,
                            act_f=lambda x: mx.sym.Activation(x, act_type="tanh"),
                            i2h_weight=i2h_weight, i2h_bias=i2h_bias,
                            h2h_weight=h2h_weight, h2h_bias=h2h_bias, name=name)

def step_lstm(num_hidden, data, prev_h, prev_c, i2h_weight, i2h_bias, h2h_weight, h2h_bias, name):
    i2h = mx.sym.FullyConnected(data=data,
                                weight=i2h_weight,
                                bias=i2h_bias,
                                num_hidden=num_hidden * 4,
                                name=name + ":i2h")
    h2h = mx.sym.FullyConnected(data=prev_h,
                                weight=h2h_weight,
                                bias=h2h_bias,
                                num_hidden=num_hidden * 4,
                                name=name + ":h2h")
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4, axis=1)
    input_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid",
                                   name=name + ":gi")
    forget_gate = mx.sym.Activation(slice_gates[1], act_type="sigmoid",
                                    name=name + ":gf")
    new_mem = mx.sym.Activation(slice_gates[2], act_type="tanh",
                                name=name + ":new_mem")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid",
                                 name=name + ":go")
    new_c = forget_gate * prev_c + input_gate * new_mem
    new_h = out_gate * mx.sym.Activation(new_c, act_type="tanh")
    return new_h, new_c

def step_gru(num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight, h2h_bias, name):
    i2h = mx.sym.FullyConnected(data=data,
                                weight=i2h_weight,
                                bias=i2h_bias,
                                num_hidden=num_hidden * 3,
                                name=name + ":i2h")
    h2h = mx.sym.FullyConnected(data=prev_h,
                                weight=h2h_weight,
                                bias=h2h_bias,
                                num_hidden=num_hidden * 3,
                                name=name + ":h2h")
    i2h_slice = mx.sym.SliceChannel(i2h, num_outputs=3, axis=1)
    h2h_slice = mx.sym.SliceChannel(h2h, num_outputs=3, axis=1)
    reset_gate = mx.sym.Activation(i2h_slice[0] + h2h_slice[0], act_type="sigmoid",
                                   name=name + ":gr")
    update_gate = mx.sym.Activation(i2h_slice[1] + h2h_slice[1], act_type="sigmoid",
                                    name=name + ":gu")
    new_mem = mx.sym.Activation(i2h_slice[2] + reset_gate * h2h_slice[2], act_type="tanh",
                                name=name + ":new_mem")
    new_h = update_gate * prev_h + (1 - update_gate) * new_mem
    return new_h

def LSTM(seq_len, layer_num, num_hidden, data, prev_h, prev_c, i2h_weight, i2h_bias, h2h_weight, h2h_bias, name):
    data = mx.sym.SliceChannel(data, num_outputs=seq_len, axis=0, squeeze_axis=True)
    prev_h = mx.sym.SliceChannel(prev_h, num_outputs=layer_num, axis=0, squeeze_axis=True)
    prev_c = mx.sym.SliceChannel(prev_c, num_outputs=layer_num, axis=0, squeeze_axis=True)
    i2h_weight = mx.sym.SliceChannel(i2h_weight, num_outputs=layer_num, axis=0, squeeze_axis=True)
    i2h_bias = mx.sym.SliceChannel(i2h_bias, num_outputs=layer_num, axis=0, squeeze_axis=True)
    h2h_weight = mx.sym.SliceChannel(h2h_weight, num_outputs=layer_num, axis=0, squeeze_axis=True)
    h2h_bias = mx.sym.SliceChannel(h2h_bias, num_outputs=layer_num, axis=0, squeeze_axis=True)
    data = [data[i] for i in range(seq_len)]
    prev_h = [prev_h[i] for i in range(layer_num)]
    prev_c = [prev_c[i] for i in range(layer_num)]
    for l in range(layer_num):
        out_h = []
        out_c = []
        for i in range(seq_len):
            new_h, new_c = step_lstm(num_hidden=num_hidden,
                                     data=data[i],
                                     prev_h=prev_h[l],
                                     prev_c=prev_c[l],
                                     i2h_weight=i2h_weight[l], i2h_bias=i2h_bias[l],
                                     h2h_weight=h2h_weight[l], h2h_bias=h2h_bias[l], name=name)
            prev_h[l], prev_c[l] = new_h, new_c
            out_h.append(new_h)
            out_c.append(new_c)
            data[i] = new_h
    out_h = mx.sym.Reshape(data=mx.sym.Concat(*out_h, num_args=len(out_h), dim=0),
                           shape=(seq_len, -1, num_hidden))
    out_c = mx.sym.Reshape(data=mx.sym.Concat(*out_c, num_args=len(out_c), dim=0),
                           shape=(seq_len, -1, num_hidden))
    return out_h, out_c

def Tanh_RNN(seq_len, layer_num, num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight, h2h_bias, name):
    data = mx.sym.SliceChannel(data, num_outputs=seq_len, axis=0, squeeze_axis=True)
    prev_h = mx.sym.SliceChannel(prev_h, num_outputs=layer_num, axis=0, squeeze_axis=True)
    i2h_weight = mx.sym.SliceChannel(i2h_weight, num_outputs=layer_num, axis=0, squeeze_axis=True)
    i2h_bias = mx.sym.SliceChannel(i2h_bias, num_outputs=layer_num, axis=0, squeeze_axis=True)
    h2h_weight = mx.sym.SliceChannel(h2h_weight, num_outputs=layer_num, axis=0, squeeze_axis=True)
    h2h_bias = mx.sym.SliceChannel(h2h_bias, num_outputs=layer_num, axis=0, squeeze_axis=True)
    data = [data[i] for i in range(seq_len)]
    prev_h = [prev_h[i] for i in range(layer_num)]
    for l in range(layer_num):
        out_h = []
        for i in range(seq_len):
            new_h = step_tanh_rnn(num_hidden=num_hidden,
                              data=data[i],
                              prev_h=prev_h[l],
                              i2h_weight=i2h_weight[l], i2h_bias=i2h_bias[l],
                              h2h_weight=h2h_weight[l], h2h_bias=h2h_bias[l], name=name)
            prev_h[l] = new_h
            out_h.append(new_h)
            data[i] = new_h
    out_h = mx.sym.Reshape(data=mx.sym.Concat(*out_h, num_args=len(out_h), dim=0),
                           shape=(seq_len, -1, num_hidden))
    return out_h

def Tanh_RNN_numpy(seq_len, layer_num, num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight,
                   h2h_bias):
    out_h = numpy.zeros((seq_len, data.shape[1], num_hidden))
    for l in range(layer_num):
        if l > 0:
            data = out_h
        for i in range(seq_len):
            out_h[i] = numpy.tanh(data[i].dot(i2h_weight[l].T) + i2h_bias[l].reshape((1, num_hidden)) + \
                                  prev_h[l].dot(h2h_weight[l].T) + h2h_bias[l].reshape((1, num_hidden)))
            prev_h[l] = out_h[i]
            print out_h
    return out_h

def get_cudnn_parameters(i2h_weight, i2h_bias, h2h_weight, h2h_bias):
    """Get a single param symbol for a CuDNN RNN layer based on the given parameters

    Parameters
    ----------
    i2h_weight
    i2h_bias
    h2h_weight
    h2h_bias

    Returns
    -------
    """
    return mx.sym.Concat(mx.sym.Reshape(data=i2h_weight, shape=(-1,)),
            mx.sym.Reshape(data=h2h_weight, shape=(-1,)),
            mx.sym.Reshape(data=i2h_bias, shape=(-1,)),
            mx.sym.Reshape(data=h2h_bias, shape=(-1,)), num_args=4, dim=0)

def compare_lstm():
    data = mx.sym.Variable('data')
    i2h_weight = mx.sym.Variable('i2h_weight')
    i2h_bias = mx.sym.Variable('i2h_bias')
    h2h_weight = mx.sym.Variable('h2h_weight')
    h2h_bias = mx.sym.Variable('h2h_bias')
    init_h = mx.sym.Variable('init_h')
    init_c = mx.sym.Variable('init_c')
    seq_len = 1
    batch_size = 1
    data_dim = 1
    state_dim = 1
    layer_num = 2

    data_npy = numpy.random.standard_normal((seq_len, batch_size, data_dim))
    out_grad_npy = numpy.random.standard_normal((seq_len, batch_size, state_dim))
    i2h_weight_npy = numpy.random.standard_normal((layer_num, state_dim*4, data_dim))
    i2h_bias_npy = numpy.random.standard_normal((layer_num, state_dim*4))
    h2h_weight_npy = numpy.random.standard_normal((layer_num, state_dim*4, state_dim))
    h2h_bias_npy = numpy.random.standard_normal((layer_num, state_dim*4))
    init_h_npy = numpy.random.standard_normal((layer_num, batch_size, state_dim))
    init_c_npy = numpy.random.standard_normal((layer_num, batch_size, state_dim))


    # i2h_weight_split = mx.sym.SliceChannel(i2h_weight, num_outputs=layer_num, squeeze_axis=True)
    # i2h_bias_split = mx.sym.SliceChannel(i2h_bias, num_outputs=layer_num, squeeze_axis=True)
    # h2h_weight_split = mx.sym.SliceChannel(h2h_weight, num_outputs=layer_num, squeeze_axis=True)
    # h2h_bias_split = mx.sym.SliceChannel(h2h_bias, num_outputs=layer_num, squeeze_axis=True)
    #
    #
    # lstm_parameters = mx.sym.Concat(*(sum([[mx.sym.Reshape(data=i2h_weight_split[i], shape =(-1,)),
    #                                 mx.sym.Reshape(data=h2h_weight_split[i], shape =(-1,))] for i in range(layer_num)], []) +
    #                                   sum([[mx.sym.Reshape(data=i2h_bias_split[i], shape =(-1,)),
    #                                 mx.sym.Reshape(data=h2h_bias_split[i], shape =(-1,))] for i in range(layer_num)], [])),
    #                                 num_args=4*layer_num, dim=0)
    i2h_weight_trans = mx.sym.transpose(i2h_weight, axes=(1, 2, 0))
    i2h_bias_trans = mx.sym.transpose(i2h_bias, axes=(0, 1))
    h2h_weight_trans = mx.sym.transpose(h2h_weight, axes=(1, 2, 0))
    h2h_bias_trans = mx.sym.transpose(h2h_bias, axes=(0, 1))
    lstm_parameters = mx.sym.Concat(mx.sym.Reshape(data=i2h_weight_trans, shape =(-1,)),
                                    mx.sym.Reshape(data=h2h_weight_trans, shape =(-1,)),
                                    mx.sym.Reshape(data=i2h_bias_trans, shape=(-1,)),
                                    mx.sym.Reshape(data=h2h_bias_trans, shape=(-1,)),
                                    num_args=4, dim=0)

    rnn = mx.sym.RNN(data=data, state_size=state_dim, num_layers=layer_num,
                     parameters=lstm_parameters,
                     mode="lstm",
                     state=init_h, state_cell=init_c, name="LSTM",
                     state_outputs=True)
    rnn = mx.sym.Group([rnn[0], mx.sym.BlockGrad(rnn[1]), mx.sym.BlockGrad(rnn[2])])

    rnn_for_checking = mx.sym.RNN(data=data, state_size=state_dim, num_layers=layer_num,
                                  parameters=lstm_parameters,
                                  mode="lstm",
                                  state=init_h, state_cell=init_c, name="LSTM",
                                  state_outputs=False)

    # check_numeric_gradient(sym=rnn_for_checking,
    #                        ctx=mx.gpu(),
    #                        locations={'data': data_npy, 'init_h': init_h_npy, 'init_c' : init_c_npy,
    #                        'i2h_weight' : i2h_weight_npy, 'h2h_weight' : h2h_weight_npy,
    #                        'i2h_bias' : i2h_bias_npy, 'h2h_bias' : h2h_bias_npy},
    #                        grad_nodes=['i2h_weight', 'h2h_weight', 'i2h_bias', 'h2h_bias'], check_eps=1E-2)
    #
    # ch = raw_input()
    exe = rnn_for_checking.simple_bind(ctx=mx.gpu(), data=(seq_len, batch_size, data_dim),
                          init_h=(layer_num, batch_size, state_dim),
                          init_c=(layer_num, batch_size, state_dim),
                          i2h_weight=i2h_weight_npy.shape,
                          h2h_weight=h2h_weight_npy.shape,
                          i2h_bias=i2h_bias_npy.shape,
                          h2h_bias=h2h_bias_npy.shape)
    for k, v in exe.arg_dict.items():
        print(k, v.shape)

    outputs = exe.forward(is_train=True, data=data_npy, init_h=init_h_npy, init_c=init_c_npy,
                          i2h_weight=i2h_weight_npy, h2h_weight=h2h_weight_npy,
                          i2h_bias=i2h_bias_npy, h2h_bias=h2h_bias_npy)
    exe.backward(out_grads=[nd.array(out_grad_npy, ctx=mx.gpu())])
    print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    # for k, v in exe.grad_dict.items():
    #     print(k, v.asnumpy())
    o1 = outputs[0].asnumpy()
    #c_last1 = outputs[2].asnumpy()

    rnn2 = LSTM(num_hidden=state_dim, layer_num=layer_num,
                seq_len=seq_len, data=data, prev_h=init_h, prev_c=init_c,
                i2h_weight=i2h_weight, i2h_bias=i2h_bias, h2h_weight=h2h_weight, h2h_bias=h2h_bias,
                name="LSTM2")

    rnn2 = mx.sym.Group([rnn2[0], mx.sym.BlockGrad(rnn2[1])])
    # check_numeric_gradient(sym=rnn2[0],
    #                        ctx=mx.gpu(),
    #                        locations={'data': data_npy, 'init_h': init_h_npy, 'init_c' : init_c_npy,
    #                        'i2h_weight' : i2h_weight_npy, 'h2h_weight' : h2h_weight_npy,
    #                        'i2h_bias' : i2h_bias_npy, 'h2h_bias' : h2h_bias_npy},
    #                        grad_nodes=['h2h_weight', 'i2h_weight', 'h2h_weight', 'i2h_bias', 'h2h_bias'], check_eps=1E-2)
    #
    # ch = raw_input()
    exe2 = rnn2.simple_bind(ctx=mx.gpu(), data=(seq_len, batch_size, data_dim),
                            init_h=(layer_num, batch_size, state_dim),
                            init_c=(layer_num, batch_size, state_dim),
                            i2h_weight=i2h_weight_npy.shape,
                            h2h_weight=h2h_weight_npy.shape,
                            i2h_bias=i2h_bias_npy.shape,
                            h2h_bias=h2h_bias_npy.shape)
    for k, v in exe2.arg_dict.items():
        print(k, v.shape)

    exe2.arg_dict['i2h_weight'][:] = i2h_weight_npy
    exe2.arg_dict['i2h_bias'][:] = i2h_bias_npy
    exe2.arg_dict['h2h_weight'][:] = h2h_weight_npy
    exe2.arg_dict['h2h_bias'][:] = h2h_bias_npy
    outputs2 = exe2.forward(is_train=True, data=data_npy, init_h=init_h_npy, init_c=init_c_npy)
    exe2.backward(out_grads=[nd.array(out_grad_npy, ctx=mx.gpu())])
    # for k, v in exe2.grad_dict.items():
    #     print(k, v.asnumpy())
    print(outputs2[0].shape)
    print(outputs2[1].shape)
    o2 = outputs2[0].asnumpy()
    #c_last2 = outputs2[1].asnumpy()[-1:, :, :]
    print(numpy.square(o2-o1).sum())
    #print(numpy.square(c_last2 - c_last1).sum())

    for k in exe.grad_dict:
        print(k, exe.grad_dict[k].asnumpy(), exe2.grad_dict[k].asnumpy())
        print(k, numpy.square(exe.grad_dict[k].asnumpy()-exe2.grad_dict[k].asnumpy()).sum())
        #print(k, numpy.square(exe.grad_dict[k].asnumpy() - exe2.grad_dict[k].asnumpy()).sum())

def compare_tanh_rnn():
    data = mx.sym.Variable('data')
    i2h_weight = mx.sym.Variable('i2h_weight')
    i2h_bias = mx.sym.Variable('i2h_bias')
    h2h_weight = mx.sym.Variable('h2h_weight')
    h2h_bias = mx.sym.Variable('h2h_bias')
    init_h = mx.sym.Variable('init_h')
    seq_len = 5
    batch_size = 8
    data_dim = 4
    state_dim = 4
    layer_num = 5

    data_npy = numpy.random.standard_normal((seq_len, batch_size, data_dim))
    out_grad_npy = numpy.random.standard_normal((seq_len, batch_size, state_dim))
    i2h_weight_npy = numpy.random.standard_normal((layer_num, state_dim, data_dim))
    i2h_bias_npy = numpy.random.standard_normal((layer_num, state_dim))
    h2h_weight_npy = numpy.random.standard_normal((layer_num, state_dim, state_dim))
    h2h_bias_npy = numpy.random.standard_normal((layer_num, state_dim))
    init_h_npy = numpy.random.standard_normal((layer_num, batch_size, state_dim))

    # i2h_weight_split = mx.sym.SliceChannel(i2h_weight, num_outputs=layer_num, squeeze_axis=True)
    # i2h_bias_split = mx.sym.SliceChannel(i2h_bias, num_outputs=layer_num, squeeze_axis=True)
    # h2h_weight_split = mx.sym.SliceChannel(h2h_weight, num_outputs=layer_num, squeeze_axis=True)
    # h2h_bias_split = mx.sym.SliceChannel(h2h_bias, num_outputs=layer_num, squeeze_axis=True)
    #
    #
    # lstm_parameters = mx.sym.Concat(*(sum([[mx.sym.Reshape(data=i2h_weight_split[i], shape =(-1,)),
    #                                 mx.sym.Reshape(data=h2h_weight_split[i], shape =(-1,))] for i in range(layer_num)], []) +
    #                                   sum([[mx.sym.Reshape(data=i2h_bias_split[i], shape =(-1,)),
    #                                 mx.sym.Reshape(data=h2h_bias_split[i], shape =(-1,))] for i in range(layer_num)], [])),
    #                                 num_args=4*layer_num, dim=0)


    # i2h_weight_trans = mx.sym.transpose(i2h_weight, axes=(0, 1, 2))
    # i2h_bias_trans = mx.sym.transpose(i2h_bias, axes=(0, 1))
    # h2h_weight_trans = mx.sym.transpose(h2h_weight, axes=(0, 1, 2))
    # h2h_bias_trans = mx.sym.transpose(h2h_bias, axes=(0, 1))
    # lstm_parameters = mx.sym.Concat(mx.sym.Reshape(data=i2h_weight_trans, shape=(-1,)),
    #                                 mx.sym.Reshape(data=h2h_weight_trans, shape=(-1,)),
    #                                 mx.sym.Reshape(data=i2h_bias_trans, shape=(-1,)),
    #                                 mx.sym.Reshape(data=h2h_bias_trans, shape=(-1,)),
    #                                 num_args=4, dim=0)
    #
    # rnn_for_checking = mx.sym.RNN(data=data, state_size=state_dim, num_layers=layer_num,
    #                               parameters=lstm_parameters,
    #                               mode="rnn_tanh",
    #                               state=init_h, name="RNN_Tanh",
    #                               state_outputs=False)


    init_h_split = mx.sym.SliceChannel(init_h, num_outputs=layer_num, axis=0, squeeze_axis=False)
    i2h_weight_split = mx.sym.SliceChannel(i2h_weight, num_outputs=layer_num, axis=0, squeeze_axis=True)
    i2h_bias_split = mx.sym.SliceChannel(i2h_bias, num_outputs=layer_num, axis=0, squeeze_axis=True)
    h2h_weight_split = mx.sym.SliceChannel(h2h_weight, num_outputs=layer_num, axis=0, squeeze_axis=True)
    h2h_bias_split = mx.sym.SliceChannel(h2h_bias, num_outputs=layer_num, axis=0, squeeze_axis=True)
    rnn_for_checking = None
    for i in range(layer_num):
        if i == 0:
            rnn_for_checking = mx.sym.RNN(data=data, state_size=state_dim, num_layers=1,
                                          parameters=get_cudnn_parameters(i2h_weight=i2h_weight_split[i],
                                                                          h2h_weight=h2h_weight_split[i],
                                                                          i2h_bias=i2h_bias_split[i],
                                                                          h2h_bias=h2h_bias_split[i]),
                                          mode="rnn_tanh",
                                          state=init_h_split[i],
                                          name="RNN_Tanh",
                                          state_outputs=False)
        else:
            rnn_for_checking = mx.sym.RNN(data=rnn_for_checking, state_size=state_dim, num_layers=1,
                                          parameters=get_cudnn_parameters(
                                              i2h_weight=i2h_weight_split[i],
                                              h2h_weight=h2h_weight_split[i],
                                              i2h_bias=i2h_bias_split[i],
                                              h2h_bias=h2h_bias_split[i]),
                                          mode="rnn_tanh",
                                          state=init_h_split[i],
                                          name="RNN_Tanh",
                                          state_outputs=False)
    # check_numeric_gradient(sym=rnn_for_checking,
    #                        ctx=mx.gpu(),
    #                        locations={'data': data_npy, 'init_h': init_h_npy,# 'init_c' : init_c_npy,
    #                        'i2h_weight' : i2h_weight_npy, 'h2h_weight' : h2h_weight_npy,
    #                        'i2h_bias' : i2h_bias_npy, 'h2h_bias' : h2h_bias_npy},
    #                        grad_nodes=['data', 'h2h_weight', 'i2h_bias', 'h2h_bias'], check_eps=1E-2)
    #
    # ch = raw_input()
    exe = rnn_for_checking.simple_bind(ctx=mx.gpu(), data=(seq_len, batch_size, data_dim),
                                       init_h=(layer_num, batch_size, state_dim),
                                       i2h_weight=i2h_weight_npy.shape,
                                       h2h_weight=h2h_weight_npy.shape,
                                       i2h_bias=i2h_bias_npy.shape,
                                       h2h_bias=h2h_bias_npy.shape)
    for k, v in exe.arg_dict.items():
        print(k, v.shape)

    outputs = exe.forward(is_train=True, data=data_npy, init_h=init_h_npy,
                          i2h_weight=i2h_weight_npy, h2h_weight=h2h_weight_npy,
                          i2h_bias=i2h_bias_npy, h2h_bias=h2h_bias_npy)
    exe.backward(out_grads=[nd.array(out_grad_npy, ctx=mx.gpu())])
    print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # for k, v in exe.grad_dict.items():
    #     print(k, v.asnumpy())
    o1 = outputs[0].asnumpy()
    # c_last1 = outputs[2].asnumpy()

    rnn2 = Tanh_RNN(num_hidden=state_dim, layer_num=layer_num,
                seq_len=seq_len, data=data, prev_h=init_h,
                i2h_weight=i2h_weight, i2h_bias=i2h_bias, h2h_weight=h2h_weight, h2h_bias=h2h_bias,
                name="RNN_Tanh2")

    # check_numeric_gradient(sym=rnn2,
    #                        ctx=mx.gpu(),
    #                        locations={'data': data_npy, 'init_h': init_h_npy,
    #                        'i2h_weight' : i2h_weight_npy, 'h2h_weight' : h2h_weight_npy,
    #                        'i2h_bias' : i2h_bias_npy, 'h2h_bias' : h2h_bias_npy},
    #                        grad_nodes=['h2h_weight', 'i2h_weight', 'h2h_weight', 'i2h_bias', 'h2h_bias'], check_eps=1E-2)
    #
    # ch = raw_input()
    exe2 = rnn2.simple_bind(ctx=mx.gpu(), data=(seq_len, batch_size, data_dim),
                            init_h=(layer_num, batch_size, state_dim),
                            i2h_weight=i2h_weight_npy.shape,
                            h2h_weight=h2h_weight_npy.shape,
                            i2h_bias=i2h_bias_npy.shape,
                            h2h_bias=h2h_bias_npy.shape)
    for k, v in exe2.arg_dict.items():
        print(k, v.shape)

    exe2.arg_dict['i2h_weight'][:] = i2h_weight_npy
    exe2.arg_dict['i2h_bias'][:] = i2h_bias_npy
    exe2.arg_dict['h2h_weight'][:] = h2h_weight_npy
    exe2.arg_dict['h2h_bias'][:] = h2h_bias_npy
    outputs2 = exe2.forward(is_train=True, data=data_npy, init_h=init_h_npy)
    exe2.backward(out_grads=[nd.array(out_grad_npy, ctx=mx.gpu())])
    # for k, v in exe2.grad_dict.items():
    #     print(k, v.asnumpy())
    print(outputs2[0].shape)
    o2 = outputs2[0].asnumpy()
    # c_last2 = outputs2[1].asnumpy()[-1:, :, :]
    print("data", data_npy)
    print("init_h", init_h_npy)
    print("i2h_weight", i2h_weight_npy)
    print("i2h_bias", i2h_bias_npy)
    print("h2h_weight", h2h_weight_npy)
    print("h2h_bias", h2h_bias_npy)
    print("o1", o1)
    print("o2", o2)
    numpy_output = Tanh_RNN_numpy(seq_len=seq_len, layer_num=layer_num, num_hidden=state_dim, data=data_npy,
                                  prev_h=init_h_npy, i2h_weight=i2h_weight_npy, i2h_bias=i2h_bias_npy,
                                  h2h_weight=h2h_weight_npy, h2h_bias=h2h_bias_npy)
    print("numpy_output", numpy_output)
    print(numpy.square(o2 - o1).sum())
    # print(numpy.square(c_last2 - c_last1).sum())

    for k in exe.grad_dict:
        #print(k, exe.grad_dict[k].asnumpy(), exe2.grad_dict[k].asnumpy())
        print(k, numpy.square(exe.grad_dict[k].asnumpy() - exe2.grad_dict[k].asnumpy()).mean())
def test_RNN_class(typ="lstm", ret_typ="out"):
    num_hidden = (128, 128, 128)
    data_dim = 512
    seq_len = 20
    minibatch_size = 8
    data_npy = numpy.random.standard_normal((seq_len, minibatch_size, data_dim))
    init_h_0 = numpy.random.standard_normal((minibatch_size, num_hidden[0]))
    init_h_1 = numpy.random.standard_normal((minibatch_size, num_hidden[1]))
    init_h_2 = numpy.random.standard_normal((minibatch_size, num_hidden[2]))

    if typ == "lstm":
        init_c_0 = numpy.random.standard_normal((minibatch_size, num_hidden[0]))
        init_c_1 = numpy.random.standard_normal((minibatch_size, num_hidden[1]))
        init_c_2 = numpy.random.standard_normal((minibatch_size, num_hidden[2]))
    param_shapes = get_rnn_param_shapes(num_hidden=num_hidden, data_dim=data_dim, typ=typ)
    i2h_weight_npy = [numpy.random.standard_normal(s) for s in param_shapes["i2h_weight"]]
    i2h_bias_npy = [numpy.random.standard_normal(s)/100 for s in param_shapes["i2h_bias"]]
    h2h_weight_npy = [numpy.random.standard_normal(s) for s in param_shapes["h2h_weight"]]
    h2h_bias_npy = [numpy.random.standard_normal(s)/100 for s in param_shapes["h2h_bias"]]
    if ret_typ == "out":
        out_grad_npy = numpy.random.standard_normal((seq_len, minibatch_size, num_hidden[2]))
    elif ret_typ == "state":
        mult = 2 if typ == "lstm" else 1
        out_grad_npy = numpy.random.standard_normal((minibatch_size, mult * (num_hidden[0] + num_hidden[1] + num_hidden[2])))
    data = mx.sym.Variable("data")
    rnn = RNN(data_dim=data_dim, num_hidden=num_hidden, typ=typ, name=typ.upper())
    rnn_cudnn = RNN(data_dim=data_dim, num_hidden=num_hidden, typ=typ, cudnn_opt=True, name=typ.upper()+"-cudnn")
    if ret_typ == "state":
        if typ == "lstm":
            rnn_out_h, rnn_out_c = rnn.step(data=data, seq_len=seq_len, ret_typ=ret_typ)
            rnn_cudnn_out_h, rnn_cudnn_out_c = rnn_cudnn.step(data=data, seq_len=seq_len, ret_typ=ret_typ)
            rnn_sym = mx.sym.Concat(*(rnn_out_h + rnn_out_c), num_args=len(num_hidden) * 2, dim=1)
            rnn_cudnn_sym = mx.sym.Concat(*(rnn_cudnn_out_h + rnn_cudnn_out_c), num_args=len(num_hidden) * 2, dim=1)
        else:
            rnn_out_h = rnn.step(data=data, seq_len=seq_len, ret_typ=ret_typ)
            rnn_cudnn_out_h = rnn_cudnn.step(data=data, seq_len=seq_len, ret_typ=ret_typ)
            rnn_sym = mx.sym.Concat(*(rnn_out_h), num_args=len(num_hidden), dim=1)
            rnn_cudnn_sym = mx.sym.Concat(*(rnn_cudnn_out_h), num_args=len(num_hidden), dim=1)
    else:
        rnn_out_h = rnn.step(data=data, seq_len=seq_len, ret_typ=ret_typ)
        rnn_cudnn_out_h = rnn_cudnn.step(data=data, seq_len=seq_len, ret_typ=ret_typ)
        rnn_sym = rnn_out_h[-1]
        rnn_cudnn_sym = rnn_cudnn_out_h[-1]

    if typ == "lstm":
        rnn_exe = rnn_sym.simple_bind(ctx=mx.gpu(),
                                            **{'data':(seq_len, minibatch_size, data_dim),
                                               rnn.name + '->layer0:init_h': (minibatch_size, num_hidden[0]),
                                               rnn.name + '->layer0:init_c': (minibatch_size, num_hidden[0]),
                                               rnn.name + '->layer1:init_h': (minibatch_size, num_hidden[1]),
                                               rnn.name + '->layer1:init_c': (minibatch_size, num_hidden[1]),
                                               rnn.name + '->layer2:init_h': (minibatch_size, num_hidden[2]),
                                               rnn.name + '->layer2:init_c': (minibatch_size, num_hidden[2])})
        rnn_cudnn_exe = rnn_cudnn_sym.simple_bind(ctx=mx.gpu(),
                                                        **{'data': (seq_len, minibatch_size, data_dim),
                                                           rnn_cudnn.name + '->layer0:init_h': (
                                                           minibatch_size, num_hidden[0]),
                                                           rnn_cudnn.name + '->layer0:init_c': (
                                                           minibatch_size, num_hidden[0]),
                                                           rnn_cudnn.name + '->layer1:init_h': (
                                                           minibatch_size, num_hidden[1]),
                                                           rnn_cudnn.name + '->layer1:init_c': (
                                                           minibatch_size, num_hidden[1]),
                                                           rnn_cudnn.name + '->layer2:init_h': (
                                                           minibatch_size, num_hidden[2]),
                                                           rnn_cudnn.name + '->layer2:init_c': (
                                                           minibatch_size, num_hidden[2])})
    else:
        rnn_exe = rnn_sym.simple_bind(ctx=mx.gpu(),
                                            **{'data': (seq_len, minibatch_size, data_dim),
                                               rnn.name + '->layer0:init_h': (
                                               minibatch_size, num_hidden[0]),
                                               rnn.name + '->layer1:init_h': (
                                               minibatch_size, num_hidden[1]),
                                               rnn.name + '->layer2:init_h': (
                                               minibatch_size, num_hidden[2])})
        rnn_cudnn_exe = rnn_cudnn_sym.simple_bind(ctx=mx.gpu(),
                                                        **{'data': (
                                                        seq_len, minibatch_size, data_dim),
                                                           rnn_cudnn.name + '->layer0:init_h': (
                                                               minibatch_size, num_hidden[0]),
                                                           rnn_cudnn.name + '->layer1:init_h': (
                                                               minibatch_size, num_hidden[1]),
                                                           rnn_cudnn.name + '->layer2:init_h': (
                                                               minibatch_size, num_hidden[2])})
    for i in range(len(num_hidden)):
        rnn_exe.arg_dict[rnn.name + '->layer%d:i2h_weight' % i][:] = i2h_weight_npy[i]
        rnn_exe.arg_dict[rnn.name + '->layer%d:h2h_weight' % i][:] = h2h_weight_npy[i]
        rnn_exe.arg_dict[rnn.name + '->layer%d:i2h_bias' % i][:] = i2h_bias_npy[i]
        rnn_exe.arg_dict[rnn.name + '->layer%d:h2h_bias' % i][:] = h2h_bias_npy[i]
        rnn_cudnn_exe.arg_dict[rnn_cudnn.name + '->layer%d:i2h_weight' %i][:] = i2h_weight_npy[i]
        rnn_cudnn_exe.arg_dict[rnn_cudnn.name + '->layer%d:h2h_weight' %i][:] = h2h_weight_npy[i]
        rnn_cudnn_exe.arg_dict[rnn_cudnn.name + '->layer%d:i2h_bias' %i][:] = i2h_bias_npy[i]
        rnn_cudnn_exe.arg_dict[rnn_cudnn.name + '->layer%d:h2h_bias' %i][:] = h2h_bias_npy[i]
    N = 1
    if typ == "lstm":
        start = time.time()
        for j in range(N):
            rnn_outputs = rnn_exe.forward(is_train=True, **{"data": data_npy,
                                                            rnn.name + '->layer0:init_h': init_h_0,
                                                            rnn.name + '->layer0:init_c': init_c_0,
                                                            rnn.name + '->layer1:init_h': init_h_1,
                                                            rnn.name + '->layer1:init_c': init_c_1,
                                                            rnn.name + '->layer2:init_h': init_h_2,
                                                            rnn.name + '->layer2:init_c': init_c_2})

            rnn_exe.backward(out_grads=[mx.nd.array(out_grad_npy, ctx=mx.gpu())])
            nd.waitall()
        end = time.time()
        print("MXNet %s Time: %g ms" %(typ.upper(), (end -start)/ N * 1000))
        start = time.time()
        for j in range(N):
            rnn_cudnn_outputs = rnn_cudnn_exe.forward(is_train=True, **{"data": data_npy,
                                                            rnn_cudnn.name + '->layer0:init_h': init_h_0,
                                                            rnn_cudnn.name + '->layer0:init_c': init_c_0,
                                                            rnn_cudnn.name + '->layer1:init_h': init_h_1,
                                                            rnn_cudnn.name + '->layer1:init_c': init_c_1,
                                                            rnn_cudnn.name + '->layer2:init_h': init_h_2,
                                                            rnn_cudnn.name + '->layer2:init_c': init_c_2})
            rnn_cudnn_exe.backward(out_grads=[mx.nd.array(out_grad_npy, ctx=mx.gpu())])
            nd.waitall()
        end = time.time()
        print("CuDNN %s Time: %g ms" % (typ.upper(), (end - start) / N * 1000))
    else:
        start = time.time()
        for j in range(N):
            rnn_outputs = rnn_exe.forward(is_train=True, **{"data": data_npy,
                                                            rnn.name + '->layer0:init_h': init_h_0,
                                                            rnn.name + '->layer1:init_h': init_h_1,
                                                            rnn.name + '->layer2:init_h': init_h_2})
            rnn_exe.backward(out_grads=[mx.nd.array(out_grad_npy, ctx=mx.gpu())])
            nd.waitall()
        end = time.time()
        print("MXNet %s Time: %g ms" %(typ.upper(), (end -start)/ N * 1000))
        start = time.time()
        for j in range(N):
            rnn_cudnn_outputs = rnn_cudnn_exe.forward(is_train=True, **{"data": data_npy,
                                                                        rnn_cudnn.name + '->layer0:init_h': init_h_0,
                                                                        rnn_cudnn.name + '->layer1:init_h': init_h_1,
                                                                        rnn_cudnn.name + '->layer2:init_h': init_h_2})
            rnn_cudnn_exe.backward(out_grads=[mx.nd.array(out_grad_npy, ctx=mx.gpu())])
            nd.waitall()
        end = time.time()
        print("CuDNN %s Time: %g ms" % (typ.upper(), (end - start) / N * 1000))
    print(numpy.square(rnn_outputs[0].asnumpy() - rnn_cudnn_outputs[0].asnumpy()).mean())
    for k, v in rnn_exe.grad_dict.items():
        if k == 'data':
            #numpy.testing.assert_allclose(v.asnumpy(), rnn_cudnn_exe.grad_dict[k].asnumpy())
            print(k, reldiff(v.asnumpy(), rnn_cudnn_exe.grad_dict[k].asnumpy()))
        else:
            postfix = k[k.find("->"):]
            #numpy.testing.assert_allclose(v.asnumpy(), rnn_cudnn_exe.grad_dict[rnn_cudnn.name + postfix].asnumpy())
            print(k, reldiff(v.asnumpy(), rnn_cudnn_exe.grad_dict[rnn_cudnn.name + postfix].asnumpy()))
#compare_tanh_rnn()
print("Testing LSTM")
test_RNN_class(typ="lstm", ret_typ="out")
print("Testing GRU")
test_RNN_class(typ="gru", ret_typ="out")
print("Testing Tanh RNN")
test_RNN_class(typ="rnn_tanh", ret_typ="out")
print("Testing Relu RNN")
test_RNN_class(typ="rnn_relu", ret_typ="out")
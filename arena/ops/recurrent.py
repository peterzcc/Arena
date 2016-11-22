from __future__ import absolute_import, division, print_function

import mxnet as mx
from ..utils import *


def step_vanilla_rnn(num_hidden, data, prev_h, act_f,
                     i2h_weight, i2h_bias, h2h_weight, h2h_bias, dropout, seq_len, name):
    data = mx.sym.Reshape(data, shape=(-1, 0), reverse=True)
    if dropout > 0.:
        data = mx.sym.Dropout(data=data, p=dropout, name=name + ":dropout")
    i2h = mx.sym.FullyConnected(data=data,
                                weight=i2h_weight,
                                bias=i2h_bias,
                                num_hidden=num_hidden)
    i2h = mx.sym.SliceChannel(i2h, num_outputs=seq_len, axis=0, name=name + ":i2h")
    all_h = []
    for i in range(seq_len):
        h2h = mx.sym.FullyConnected(data=prev_h,
                                    weight=h2h_weight,
                                    bias=h2h_bias,
                                    num_hidden=num_hidden,
                                    name=name + ":h2h")
        new_h = act_f(i2h[i] + h2h)
        all_h.append(new_h)
        prev_h = new_h
    return mx.sym.Reshape(mx.sym.Concat(*all_h, num_args=len(all_h), dim=0),
                          shape=(seq_len, -1, 0), reverse=True), all_h[-1]


def step_relu_rnn(num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight, h2h_bias,
                  dropout=0., seq_len=1, name="relu_rnn"):
    return step_vanilla_rnn(num_hidden=num_hidden, data=data, prev_h=prev_h,
                            act_f=lambda x: mx.sym.Activation(x, act_type="relu"),
                            i2h_weight=i2h_weight, i2h_bias=i2h_bias,
                            h2h_weight=h2h_weight, h2h_bias=h2h_bias,
                            dropout=dropout, seq_len=seq_len, name=name)


def step_tanh_rnn(num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight, h2h_bias,
                  dropout=0., seq_len=1, name="tanh_rnn"):
    return step_vanilla_rnn(num_hidden=num_hidden, data=data, prev_h=prev_h,
                            act_f=lambda x: mx.sym.Activation(x, act_type="tanh"),
                            i2h_weight=i2h_weight, i2h_bias=i2h_bias,
                            h2h_weight=h2h_weight, h2h_bias=h2h_bias,
                            dropout=dropout, seq_len=seq_len, name=name)


def step_lstm(num_hidden, data, prev_h, prev_c, i2h_weight, i2h_bias, h2h_weight, h2h_bias,
              dropout=0., seq_len=1, name="lstm"):
    data = mx.sym.Reshape(data, shape=(-1, 0), reverse=True)
    if dropout > 0.:
        data = mx.sym.Dropout(data=data, p=dropout, name=name + ":dropout")
    i2h = mx.sym.FullyConnected(data=data,
                                weight=i2h_weight,
                                bias=i2h_bias,
                                num_hidden=num_hidden * 4)
    i2h = mx.sym.SliceChannel(i2h, num_outputs=seq_len, axis=0, name=name + ":i2h")
    all_c = []
    all_h = []
    for i in range(seq_len):
        h2h = mx.sym.FullyConnected(data=prev_h,
                                    weight=h2h_weight,
                                    bias=h2h_bias,
                                    num_hidden=num_hidden * 4,
                                    name=name + ":h2h")
        gates = i2h[i] + h2h
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
        all_h.append(new_h)
        all_c.append(new_c)
        prev_h = new_h
        prev_c = new_c
    return mx.sym.Reshape(mx.sym.Concat(*all_h, num_args=len(all_h), dim=0),
                          shape=(seq_len, -1, 0), reverse=True),\
           mx.sym.Reshape(mx.sym.Concat(*all_c, num_args=len(all_c), dim=0),
                   shape=(seq_len, -1, 0), reverse=True),\
           all_h[-1], all_c[-1]


def step_gru(num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight, h2h_bias,
             dropout=0., seq_len=1, name="gru"):
    data = mx.sym.Reshape(data, shape=(-1, 0), reverse=True)
    if dropout > 0.:
        data = mx.sym.Dropout(data=data, p=dropout, name=name + ":dropout")
    i2h = mx.sym.FullyConnected(data=data,
                                weight=i2h_weight,
                                bias=i2h_bias,
                                num_hidden=num_hidden * 3,
                                name=name + ":i2h")
    i2h = mx.sym.SliceChannel(i2h, num_outputs=seq_len, axis=0, name=name + ":i2h")
    all_h = []
    for i in range(seq_len):
        h2h = mx.sym.FullyConnected(data=prev_h,
                                    weight=h2h_weight,
                                    bias=h2h_bias,
                                    num_hidden=num_hidden * 3,
                                    name=name + ":h2h")
        i2h_slice = mx.sym.SliceChannel(i2h[i], num_outputs=3, axis=1)
        h2h_slice = mx.sym.SliceChannel(h2h, num_outputs=3, axis=1)
        reset_gate = mx.sym.Activation(i2h_slice[0] + h2h_slice[0], act_type="sigmoid",
                                       name=name + ":gr")
        update_gate = mx.sym.Activation(i2h_slice[1] + h2h_slice[1], act_type="sigmoid",
                                        name=name + ":gu")
        new_mem = mx.sym.Activation(i2h_slice[2] + reset_gate * h2h_slice[2], act_type="tanh",
                                    name=name + ":new_mem")
        new_h = update_gate * prev_h + (1 - update_gate) * new_mem
        all_h.append(new_h)
        prev_h = new_h
    return mx.sym.Reshape(mx.sym.Concat(*all_h, num_args=len(all_h), dim=0),
                          shape=(seq_len, -1, 0), reverse=True), all_h[-1]


def get_rnn_param_shapes(num_hidden, data_dim, typ):
    """

    Parameters
    ----------
    num_hidden
    data_dim
    typ

    Returns
    -------

    """
    ret = dict()
    mult = 1
    if typ == "lstm":
        mult = 4
    elif typ == "gru":
        mult = 3
    ret['i2h_weight'] = [(mult * num_hidden[0], data_dim)] +\
                        [(mult * num_hidden[i + 1], num_hidden[i])
                         for i in range(len(num_hidden) - 1)]
    ret['h2h_weight'] = [(mult * num_hidden[i], num_hidden[i]) for i in range(len(num_hidden))]
    ret['i2h_bias'] = [(mult * num_hidden[i],) for i in range(len(num_hidden))]
    ret['h2h_bias'] = [(mult * num_hidden[i],) for i in range(len(num_hidden))]
    return ret


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
                         i2h_bias,
                         h2h_bias, num_args=4, dim=0)


class RNN(object):
    """High level API for constructing stacked RNN layers.

    To use a recurrent neural network, we can first create an RNN object and use the step function
    during the symbol construction.

    Currently four types of RNN are supported and all parameters per layer are grouped into 4 matrices.
    The data layout and transition rules are similar to the RNN API in CuDNN (https://developer.nvidia.com/cudnn)
    1) ReLU RNN:
        h_t = ReLU(W_i x_t + R_i h_{t-1} + b_{W_i} + b_{R_i})

        Parameters:
            W_{i2h} = W_i
            b_{i2h} = b_{W_i}
            W_{h2h} = R_i
            b_{h2h} = b_{R_i}
    2) Tanh RNN:
        h_t = tanh(W_i x_t + R_i h_{t-1} + b_{W_i} + b_{R_i})

        Parameters:
            W_{i2h} = W_i
            b_{i2h} = b_{W_i}
            W_{h2h} = R_i
            b_{h2h} = b_{R_i}
    3) LSTM:
        i_t = \sigma(W_i x_t + R_i h_{t-1} + b_{W_i} + b_{R_i})
        f_t = \sigma(W_f x_t + R_f h_{t-1} + b_{W_f} + b_{R_f})
        o_t = \sigma(W_o x_t + R_o h_{t-1} + b_{W_o} + b_{R_o})
        c^\prime_t = tanh(W_c x_t + R_c h_{t-1} + b_{W_c} + b_{R_c})
        c_t = f_t \circ c_{t-1} + i_t \circ c^\prime_t
        h_t = o_t \circ tanh(c_t)

        Parameters: (input_gate, forget_gate, new_mem, output_gate)
            W_{i2h} = [W_i, W_f, W_c, W_o]
            b_{i2h} = [b_{W_i}, b_{W_f}, b_{W_c}, b_{W_o}]
            W_{h2h} = [R_i, R_f, R_c, R_o]
            b_{h2h} = [b_{R_i}, b_{R_f}, b_{R_c}, b_{R_o}]
    4) GRU:
        i_t = \sigma(W_i x_t + R_i h_{t-1} + b_{W_i} + b_{R_i})
        r_t = \sigma(W_r x_t + R_r h_{t-1} + b_{W_r} + b_{R_r})
        h^\prime_t = tanh(W_h x_t + r_t \circ (R_h h_{t-1} + b_{R_h}) + b_{W_h})
        h_t = (1 - i_t) \circ h^\prime_t + i_t \circ h_{t-1}

        Parameters: (reset_gate, update_gate, new_mem)
            W_{i2h} = [W_r, W_i, W_h]
            b_{i2h} = [b_{W_r}, b_{W_i}, b_{W_h}]
            W_{h2h} = [R_r, R_i, R_h]
            b_{h2h} = [b_{R_r}, b_{R_i}, b_{R_h}]
    """

    def __init__(self, num_hidden, data_dim, typ='lstm',
                 dropout=0., zoneout=0.,
                 i2h_weight=None, i2h_bias=None,
                 h2h_weight=None, h2h_bias=None,
                 init_h=None, init_c=None,
                 cudnn_opt=False,
                 name='LSTM'):
        """Initialization of the RNN object

        Parameters
        ----------
        num_hidden : list or tuple
            Size of the hidden state for all the layers
        data_dim : int
            Dimension of the input data to the symbol
        typ: str
            Type of the Recurrent Neural Network, can be 'gru', 'lstm', 'rnn_relu', 'rnn_tanh'
        dropout : list or tuple, optional
            Dropout ratio for all the hidden layers. Use 0 to indicate no-dropout.
        zoneout : list or tuple, optional
            Zoneout ratio for all the hidden layers. Use 0 to indicate no-zoneout.
        i2h_weight : list or tuple, optional
            Weight of the connections between the input and the hidden state.
        i2h_bias : list or tuple, optional
            Bias of the connections between the input and the hidden state.
        h2h_weight : list or tuple, optional
            Weight of the connections (including gates) between the hidden states of consecutive timestamps.
        h2h_bias : list or tuple, optional
            Bias of the connections (including gates) between the hidden states of consecutive timestamps.
        init_h : list or tuple, optional
            Initial hidden states of all the layers
        init_c : list or tuple, optional
            Initial cell states of all the layers. Only applicable when `typ` is "LSTM"
        cudnn_opt : bool, optional
            If True, the CuDNN version of RNN will be used. Also, the generated symbol could only be
            used with GPU and `zoneout` cannot be used.
        name : str
            Name of the object
        """
        self.name = name
        self.num_hidden = get_int_list(num_hidden)
        self.data_dim = data_dim
        self.dropout = get_float_list(dropout, expected_len=self.layer_num)
        self.zoneout = get_float_list(zoneout, expected_len=self.layer_num)
        self.typ = typ.lower()
        assert self.typ in ('gru', 'lstm', 'rnn_relu', 'rnn_tanh'),\
            "ops.RNN: typ=%s is currently not supported. We only support" \
            " 'gru', 'lstm', 'rnn_relu', 'rnn_tanh'." %typ
        default_shapes = get_rnn_param_shapes(num_hidden=self.num_hidden, data_dim=data_dim,
                                              typ=self.typ)
        self.i2h_weight = get_sym_list(i2h_weight,
                                       default_names=[self.name + "->layer%d:i2h_weight" % i
                                                      for i in range(self.layer_num)],
                                       default_shapes=default_shapes["i2h_weight"])
        self.i2h_bias = get_sym_list(i2h_bias,
                                     default_names=[self.name + "->layer%d:i2h_bias" % i
                                                    for i in range(self.layer_num)],
                                     default_shapes=default_shapes["i2h_bias"])
        self.h2h_weight = get_sym_list(h2h_weight,
                                       default_names=[self.name + "->layer%d:h2h_weight" % i
                                                      for i in range(self.layer_num)],
                                       default_shapes=default_shapes["h2h_weight"])
        self.h2h_bias = get_sym_list(h2h_bias,
                                     default_names=[self.name + "->layer%d:h2h_bias" % i
                                                    for i in range(self.layer_num)],
                                     default_shapes=default_shapes["h2h_bias"])
        self.init_h = get_sym_list(init_h,
                                   default_names=[self.name + "->layer%d:init_h" % i
                                                  for i in range(self.layer_num)])
        if typ == 'lstm':
            self.init_c = get_sym_list(init_c,
                                       default_names=[self.name + "->layer%d:init_c" % i
                                                      for i in range(self.layer_num)])
        else:
            assert init_c is None, "ops.RNN: init_c should only be used when `typ=lstm`"
            self.init_c = None
        self.cudnn_opt = cudnn_opt
        if self.cudnn_opt:
            assert self.zoneout == [0.] * self.layer_num

    @property
    def layer_num(self):
        return len(self.num_hidden)

    @property
    def params(self):
        return self.i2h_weight + self.i2h_bias + self.h2h_weight + self.h2h_bias

    def step(self, data, prev_h=None, prev_c=None, seq_len=1, ret_typ="all"):
        """Feed the data sequence into the RNN and get the state symbols.

        Parameters
        ----------
        data : list or tuple or Symbol
            The input data. Shape: (seq_len, batch_size, data_dim)
        prev_h : list or tuple or Symbol or None, optional
            The initial hidden states. If None, the symbol constructed during initialization
            will be used.
            Number of the initial states must be the same as the layer number,
            e.g, [h0, h1, h2] for a 3-layer RNN
        prev_c : list or tuple or Symbol or None, optional
            The initial cell states. Only applicable when `typ` is 'lstm'. If None,
            the symbol constructed during initialization will be used.
            Number of the initial states must be the same as the layer number,
            e.g, [c0, c1, c2] for a 3-layer LSTM
        seq_len : int, optional
            Length of the data sequence
        ret_typ : str, optional
            Determine the parts of the states to return, which can be 'all', 'out', 'state'
            IMPORTANT!! When `cudnn_opt` is on, only the 'out' flag is supported.
            If 'all', symbols that represent states of all the timestamps as well as
             the state of the last timestamp will be returned,
                e.g, For a 3-layer GRU and length-10 data sequence, the return value will be
                     ([h0, h1, h2], [h0_9, h1_9, h2_9])
                      Here all hi are of shape(seq_len, batch_size, num_hidden[i]) and
                      all hi_j are of shape(batch_size, num_hidden[i])
                     For a 3-layer LSTM and length-10 data sequence, the return value contains both state and cell
                     ([h0, h1, h2], [c0, c1, c2], [h0_9, h1_9, h2_9], [c0_9, c1_9, c2_9])
            If 'out', state outputs of the layers will be returned,
                e.g, For a 3-layer GRU/LSTM and length-10 data sequence, the return value will be
                     [h0, h1, h2]
            If 'state', last state/cell will be returned,
                e.g, For a 3-layer GRU and length-10 data sequence, the return value will be
                     [h0_9, h1_9, h2_9]
                     For a 3-layer LSTM and length-10 data sequence, the return value will be
                     ([h0_9, h1_9, h2_9], [c0_9, c1_9, c2_9])

        Returns
        -------
        tuple
            States generated by feeding the data sequence to the network.

            If the `return_all` flag is set, states of all the timestamps will be returned.
            Otherwise states of all the timestamps will be returned.

        """
        prev_h = self.init_h if prev_h is None else get_sym_list(prev_h)
        all_h = []
        all_c = []
        last_h = []
        last_c = []
        if self.typ == 'lstm':
            prev_c = self.init_c if prev_c is None else get_sym_list(prev_c)
        else:
            assert prev_c is None,\
                'Cell states is only applicable for LSTM, type of the RNN is %s' %self.typ
        assert seq_len > 0
        if isinstance(data, (list, tuple)):
            assert len(data) == seq_len, \
                "Data length error, expected:%d, received:%d" % (len(data), seq_len)
            data = mx.sym.Reshape(mx.sym.Concat(*data, num_args=len(data), dim=0),
                                  shape=(seq_len, -1, 0), reverse=True)
        if self.cudnn_opt:
            # Use the CuDNN version for each layer.
            assert ret_typ in ("out", ), "Only `ret_type=out` is supported " \
                                               "when CuDNN is used."
            for i in range(self.layer_num):
                if self.typ == "lstm":
                    rnn = mx.sym.RNN(data=data,
                                     state_size=self.num_hidden[i],
                                     num_layers=1,
                                     parameters=get_cudnn_parameters(i2h_weight=self.i2h_weight[i],
                                                                     h2h_weight=self.h2h_weight[i],
                                                                     i2h_bias=self.i2h_bias[i],
                                                                     h2h_bias=self.h2h_bias[i]),
                                     mode=self.typ,
                                     p=self.dropout[i],
                                     state=mx.sym.expand_dims(prev_h[i], axis=0),
                                     state_cell=mx.sym.expand_dims(prev_c[i], axis=0),
                                     name=self.name + "->layer%d" %i,
                                     state_outputs=False)
                    data = rnn
                    all_h.append(rnn)
                else:
                    rnn = mx.sym.RNN(data=data,
                                     state_size=self.num_hidden[i],
                                     num_layers=1,
                                     parameters=get_cudnn_parameters(i2h_weight=self.i2h_weight[i],
                                                                     h2h_weight=self.h2h_weight[i],
                                                                     i2h_bias=self.i2h_bias[i],
                                                                     h2h_bias=self.h2h_bias[i]),
                                     mode=self.typ,
                                     p=self.dropout[i],
                                     state=mx.sym.expand_dims(prev_h[i], axis=0),
                                     name=self.name + "->layer%d" %i,
                                     state_outputs=False)
                    data = rnn
                    all_h.append(rnn)
            if ret_typ == 'out':
                return all_h
            else:
                raise NotImplementedError
        else:
            #TODO Optimize this part by computing matrix multiplication first
            for i in range(self.layer_num):
                if self.typ == "lstm":
                    layer_all_h, layer_all_c, layer_last_h, layer_last_c =\
                        step_lstm(num_hidden=self.num_hidden[i], data=data,
                                  prev_h=prev_h[i], prev_c=prev_c[i],
                                  i2h_weight=self.i2h_weight[i], i2h_bias=self.i2h_bias[i],
                                  h2h_weight=self.h2h_weight[i], h2h_bias=self.h2h_bias[i],
                                  seq_len=seq_len,
                                  dropout=self.dropout[i],
                                  name=self.name + "->layer%d"%i)
                    all_h.append(layer_all_h)
                    all_c.append(layer_all_c)
                    last_h.append(layer_last_h)
                    last_c.append(layer_last_c)
                else:
                    step_func = None
                    if self.typ == 'rnn_tanh':
                        step_func = step_tanh_rnn
                    elif self.typ == 'rnn_relu':
                        step_func = step_relu_rnn
                    elif self.typ == 'gru':
                        step_func = step_gru
                    layer_all_h, layer_last_h = \
                        step_func(num_hidden=self.num_hidden[i], data=data,
                                  prev_h=prev_h[i],
                                  i2h_weight=self.i2h_weight[i], i2h_bias=self.i2h_bias[i],
                                  h2h_weight=self.h2h_weight[i], h2h_bias=self.h2h_bias[i],
                                  seq_len=seq_len,
                                  dropout=self.dropout[i],
                                  name=self.name + "->layer%d" % i)
                    all_h.append(layer_all_h)
                    last_h.append(layer_last_h)
                data = all_h[-1]
        if ret_typ == 'all':
            if self.typ == 'lstm':
                return all_h, all_c, last_h, last_c
            else:
                return all_h, last_h
        elif ret_typ == 'out':
            return all_h
        elif ret_typ == 'state':
            if self.typ == 'lstm':
                return last_h, last_c
            else:
                return last_h

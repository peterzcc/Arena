from __future__ import absolute_import, division, print_function

from arena.utils import *
import mxnet as mx


class LSTM(object):
    def __init__(self, num_hidden, dropout=0., zoneout=0.,
                 i2h_weight=None, i2h_bias=None,
                 h2h_bias=None, h2h_weight=None,
                 init_h=None, init_c=None,
                 skip_connection=False, name="LSTM"):
        """

        Parameters
        ----------
        num_hidden
        dropout
        zoneout
        i2h_weight
        i2h_bias
        h2h_bias
        h2h_weight
        init_h
        init_c
        skip_connection:
            http://arxiv.org/abs/1308.0850
        name
        """
        self.name = name
        self.num_hidden = get_int_list(num_hidden)
        self.dropout = get_float_list(dropout)
        self.zoneout = get_float_list(zoneout)
        self.i2h_weight = get_sym_list(i2h_weight,
                                       default_names=[self.name + "->layer%d:i2h_weight" % i
                                                      for i in range(self.layer_num)])
        self.i2h_bias = get_sym_list(i2h_bias,
                                     default_names=[self.name + "->layer%d:i2h_bias" % i
                                                    for i in range(self.layer_num)])
        self.h2h_weight = get_sym_list(h2h_weight,
                                       default_names=[self.name + "->layer%d:h2h_weight" % i
                                                      for i in range(self.layer_num)])
        self.h2h_bias = get_sym_list(h2h_bias,
                                     default_names=[self.name + "->layer%d:h2h_bias" % i
                                                    for i in range(self.layer_num)])
        self.init_h = get_sym_list(init_h,
                                   default_names=[self.name + "->layer%d:init_h" % i
                                                  for i in range(self.layer_num)])
        self.init_c = get_sym_list(init_c,
                                   default_names=[self.name + "->layer%d:init_c" % i
                                                  for i in range(self.layer_num)])
    @property
    def layer_num(self):
        return len(self.num_hidden)

    @property
    def params(self):
        return self.i2h_weight + self.i2h_bias + self.h2h_weight + self.h2h_bias

    def step(self, data, prev_h=None, prev_c=None, seq_length=1, return_all=False):
        """
        :param data: list or tuple or Symbol: Shape(seq_length, batch_size, data_dim)
        :param prev_h: list or tuple or Symbol: Shape(batch_size, state_dim)
        :param prev_c: list or tuple or Symbol: Shape(batch_size, state_dim)
        :param seq_length:
        :return: h, c: list of list [state of t1, state of t2,...]
        """
        prev_h = self.init_h if prev_h is None else get_sym_list(prev_h)
        prev_c = self.init_c if prev_c is None else get_sym_list(prev_c)
        out_h = []
        out_c = []
        assert seq_length > 0
        if isinstance(data, (list, tuple)):
            assert len(data) == seq_length,\
                "Data length error, expected:%d, received:%d" %(len(data), seq_length)
            data = list(data)
        else:
            if 1 == seq_length:
                data = [data]
            else:
                data = mx.sym.SliceChannel(data, num_outputs=seq_length, axis=0, squeeze_axis=True)
        for t in range(seq_length):
            data_in = data[t]
            step_h = []
            step_c = []
            if t > 0:
                prev_h = out_h[-1]
                prev_c = out_c[-1]
            for i in range(self.layer_num):
                if self.dropout[i] > 0.:
                    data_in = mx.sym.Dropout(data=data_in, p=self.dropout[i])
                i2h = mx.sym.FullyConnected(data=data_in,
                                         weight=self.i2h_weight[i],
                                         bias=self.i2h_bias[i],
                                         num_hidden=self.num_hidden[i] * 4,
                                         name=self.name + "->layer%d:i2h_t%d" % (i, t))
                h2h = mx.sym.FullyConnected(data=prev_h[i],
                                         weight=self.h2h_weight[i],
                                         bias=self.h2h_bias[i],
                                         num_hidden=self.num_hidden[i] * 4,
                                         name=self.name + "->layer%d:h2h_t%d" % (i, t))
                gates = i2h + h2h
                slice_gates = mx.sym.SliceChannel(gates, num_outputs=4, axis=1)
                input_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid",
                                       name=self.name + "->layer%d:gi_t%d" % (i,t))
                info = mx.sym.Activation(slice_gates[1], act_type="tanh",
                                         name=self.name + "->layer%d:info_t%d" %(i, t))
                forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid",
                                                name=self.name + "->layer%d:gf_t%d" % (i, t))
                out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid",
                                             name=self.name + "->layer%d:go_t%d" % (i, t))
                step_c.append((forget_gate * prev_c[i]) + (input_gate * info))
                step_h.append(out_gate * mx.sym.Activation(step_c[-1], act_type="tanh"))
                data_in = step_h[-1]
            out_h.append(step_h)
            out_c.append(step_c)
        if return_all:
            return out_h, out_c
        else:
            return out_h[-1], out_c[-1]
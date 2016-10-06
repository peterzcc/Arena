import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from arena import Base
from arena.ops import *
from arena.utils import *
from kvmn import KVMN

import matplotlib.pyplot as plt
from arena.helpers.visualization import *


class MODEL(object):
    def __init__(self, n_question, seqlen, batch_size,
                 q_embed_dim, q_state_dim, qa_embed_dim, qa_state_dim,
                 memory_size, memory_key_state_dim, memory_value_state_dim,
                 num_heads, name="KT"):
        self.n_question = n_question
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.q_state_dim = q_state_dim
        self.qa_embed_dim = qa_embed_dim
        self.qa_state_dim = qa_state_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.num_heads = num_heads
        self.name = name

    def sym_gen(self):
        ### TODO input variable 'q_data'
        q_data = mx.sym.Variable('q_data') # (seqlen, batch_size)
        ### TODO input variable 'qa_data'
        qa_data = mx.sym.Variable('qa_data')  # (seqlen, batch_size)
        ### TODO input variable 'target'
        target = mx.sym.Variable('target') #(seqlen, batch_size)

        ### Initialize Control Networks
        fnn_key_weight = mx.sym.Variable("fnn_key_weight")
        fnn_key_bias = mx.sym.Variable("fnn_key_bias")
        fnn_value_weight = mx.sym.Variable("fnn_value_weight")
        fnn_value_bias = mx.sym.Variable("fnn_value_bias")

        ### Initialize Memory
        ### TODO input variable 'init_memory_key'
        init_memory_key = mx.sym.Variable('init_memory_key')
        ### TODO input variable 'init_memory_value'
        init_memory_value = mx.sym.Variable('init_memory_value')
        ### TODO input variable 'KVMN->write_key_head:init_W_r_focus' / 'KVMN->write_key_head:init_W_u_focus'
        init_key_write_W_r_focus = mx.sym.Variable('KVMN->write_key_head:init_key_write_W_r_focus')
        init_key_write_W_u_focus = mx.sym.Variable('KVMN->write_key_head:init_key_write_W_u_focus')

        # init_memory_key = mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(init_memory_key, act_type="tanh"),
        #                                                          axis=0),
        #                                       shape=(self.batch_size, self.memory_size, self.memory_key_state_dim))
        init_memory_value = mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(init_memory_value, act_type="tanh"),
                                                                 axis=0),
                                              shape=(self.batch_size, self.memory_size, self.memory_value_state_dim))


        mem = KVMN(memory_size=self.memory_size,
                   memory_key_state_dim=self.memory_key_state_dim,
                   memory_value_state_dim=self.memory_value_state_dim,
                   num_heads=self.num_heads,
                   init_memory_key=init_memory_key,
                   init_memory_value=init_memory_value,
                   name="KVMN")

        controller_states = []
        key_read_focus_l = []
        value_read_focus_l = []
        value_read_content_l = []
        input_embed_l = []
        ### embedding
        q_embed_weight = mx.sym.Variable("q_embed_weight")
        q_embed_data = mx.sym.Embedding(data=q_data, input_dim=self.n_question,
                                        weight=q_embed_weight, output_dim=self.q_embed_dim, name='q_embed')
        slice_q_embed_data = mx.sym.SliceChannel(q_embed_data, num_outputs=self.seqlen+1, axis=0, squeeze_axis=True)
        qa_embed_weight = mx.sym.Variable("qa_embed_weight")
        qa_embed_data = mx.sym.Embedding(data=qa_data, input_dim=self.n_question*2,
                                         weight=qa_embed_weight, output_dim=self.qa_embed_dim, name='qa_embed')
        slice_qa_embed_data = mx.sym.SliceChannel(qa_embed_data, num_outputs=self.seqlen, axis=0, squeeze_axis=True)

        for i in range(self.seqlen):
            # OLD CODE
            # key_read_hidden_state = mx.sym.FullyConnected(data=slice_q_embed_data[i], num_hidden=self.q_state_dim,
            #                                        weight=fnn_key_weight, bias=fnn_key_bias, name="key_read_fc")
            # key_read_hidden_state = mx.sym.Activation(data=key_read_hidden_state, act_type='tanh', name="key_read_tanh")
            # END OLD CODE

            #key_read_hidden_state = mx.sym.Activation(data=slice_q_embed_data[i], act_type='tanh', name="key_read_tanh")
            key_read_hidden_state = slice_q_embed_data[i]

            key_read_focus = mem.key_read(key_read_hidden_state)
            ### TODO here only compute a write weight but not write to the key

            # Old Code
            # value_write_hidden_state = mx.sym.FullyConnected(data=slice_qa_embed_data[i], num_hidden=self.qa_state_dim,
            #                                        weight=fnn_value_weight, bias=fnn_value_bias, name="value_write_fc")
            # value_write_hidden_state = mx.sym.Activation(data=value_write_hidden_state, act_type='tanh', name="value_write_tanh")
            # Old Code
            value_write_hidden_state = slice_qa_embed_data[i]
            value_write_focus = key_read_focus
            new_memory_value, erase_signal, add_signal, _ = mem.value_write(value_write_focus, value_write_hidden_state)

            # OLD CODE
            # value_read_hidden_state = mx.sym.FullyConnected(data=slice_q_embed_data[i+1], num_hidden=self.q_state_dim,
            #                                               weight=fnn_key_weight, bias=fnn_key_bias, name="value_read_fc")
            # value_read_hidden_state = mx.sym.Activation(data=value_read_hidden_state, act_type='tanh', name="value_read_tanh")

            #value_read_hidden_state = mx.sym.Activation(data=slice_q_embed_data[i+1], act_type='tanh', name="value_read_tanh")
            value_read_hidden_state = slice_q_embed_data[i+1]

            value_read_focus = mem.key_read(value_read_hidden_state)
            read_value_content = mem.value_read(value_read_focus)

            ### save intermedium data
            controller_states.append(value_write_hidden_state)
            key_read_focus_l.append(key_read_focus)
            value_read_focus_l.append(value_read_focus)
            value_read_content_l.append(read_value_content)
            input_embed_l.append(slice_q_embed_data[i+1])

        all_read_value_content = mx.sym.Concat(*value_read_content_l, num_args=self.seqlen, dim=0)
        input_embed_content = mx.sym.Concat(*input_embed_l, num_args=self.seqlen, dim=0)
        #all_read_value_content = mx.sym.Dropout(all_read_value_content, p=0.5)
        read_content_embed = mx.sym.FullyConnected(data=mx.sym.Concat(all_read_value_content, input_embed_content, num_args=2, dim=1),
                                                   num_hidden=50, name="read_content_embed")
        read_content_embed = mx.sym.Activation(data=read_content_embed, act_type='tanh', name="read_content_embed_tanh")
        #read_content_embed = mx.sym.Dropout(read_content_embed, p=0.5)
        pred_fc_weight = mx.sym.Variable("pred_fc_weight")
        pred_fc_bias = mx.sym.Variable("pred_fc_bias")
        # pred = mx.sym.FullyConnected(data=all_read_value_content, num_hidden = 1,
        #                              weight=pred_fc_weight, bias=pred_fc_bias, name="final_fc")
        pred = mx.sym.FullyConnected(data=read_content_embed, num_hidden=1,
                                     weight=pred_fc_weight, bias=pred_fc_bias, name="final_fc")

        target = mx.sym.Reshape(data=target, shape=(-1,))
        # pred_prob
        pred_prob = logistic_regression_mask_output(data=mx.sym.Reshape(pred, shape=(-1, )), label=target,
                                                    ignore_label=-1,
                                                    name='final_pred')
        # Add regularizers
        lambda1 = 0
        lambda2 = 0
        S = mx.sym.dot(q_embed_weight, mx.sym.SwapAxis(init_memory_key, dim1=0, dim2=1))  # Shape: (question_num, memory_size)
        S = mx.sym.SoftmaxActivation(S)
        entropy_reg = lambda1 * entropy_multinomial(data=S)
        frobenius_reg = lambda2 * mx.sym.sum(mx.sym.square(S))
        entropy_reg = mx.sym.MakeLoss(entropy_reg, name='entropy_reg')
        frobenius_reg = mx.sym.MakeLoss(frobenius_reg, name='frobenius_reg')

        return mx.sym.Group([pred_prob,
                             entropy_reg,
                             frobenius_reg,
                             mx.sym.BlockGrad(S),
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*controller_states, dim=0,
                                               num_args=len(controller_states)),
                                 shape=(self.seqlen, -1, self.qa_state_dim))),
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*value_read_content_l, dim=0,
                                               num_args=len(value_read_content_l)),
                                 shape=(self.seqlen, -1, self.memory_value_state_dim))),
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*key_read_focus_l, dim=0,
                                               num_args=len(key_read_focus_l)),
                                 shape=(self.seqlen, -1, self.memory_size))),
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*value_read_focus_l, dim=0,
                                               num_args=len(value_read_focus_l)),
                                 shape=(self.seqlen, -1, self.memory_size)))
                             ])

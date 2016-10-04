import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from arena import Base
from arena.ops import *
from arena.utils import *
from lrua import KVMN

import matplotlib.pyplot as plt
from arena.helpers.visualization import *

################### python customized operator start ###################
# MXNET_CPU_WORKER_NTHREADS must be greater than 1
# for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
# We start by subclassing mxnet.operator.CustomOp
# and then override a few methods:
class BinaryEntropyLoss(mx.operator.CustomOp):
    # The forward function takes a list of input
    # and a list of output NDArrays.
    ### data:
    ###         pred  /  target         -(LOSS)->        loss
    ### (batch_size*seqlen, n_question)/
    ### (batch_size*seqlen, )           -(LOSS)-> (batch_size*seqlen, )
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        #print "x", x
        y = 1.0/(1.0+np.exp(-x))
        #print "y", y
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        #n_q = 123 # assist2009
        #n_q = 100 # assist2015
        n_q = 436 # algebra_2005_2006_train.txt
        #n_q = 1223 # STATICS
        l = in_data[1].asnumpy().ravel().astype(np.int)

        zero_index = np.flatnonzero(l == 0)
        next_label = (l-1) % n_q
        truth = (l-1) / n_q
        next_label[zero_index] = 0
        truth[zero_index] = 0

        y = out_data[0].asnumpy()
        np.asarray(y)
        grad = np.zeros( [ len(y),len(y[0]) ])
        grad[np.arange(len(l)), next_label] = y[np.arange(len(l)), next_label] - truth
        grad[zero_index,:] = 0
        self.assign(in_grad[0], req[0], mx.nd.array(grad))

@mx.operator.register("BinaryEntropyLoss")
# define input/output format by subclassing mx.operator.CustomOpProp.
class BinaryEntropyLossProp(mx.operator.CustomOpProp):
    def __init__(self):
        # Then we call our base constructor with
        #   need_top_grad=False
        # because softmax is a loss layer
        # and we don't need gradient input from layers above:
        super(BinaryEntropyLossProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']
    # Next we need to provide infer_shape to
    # declare the shape of our output/weight
    # and check the consistency of our input shapes
    def infer_shape(self, in_shape):
        data_shape = in_shape[0] # The first dim of an input/output tensor is batch size.
        label_shape = (in_shape[0][0],) # Our label is a set of integers, one for each data entry
        output_shape = in_shape[0] # Our output has the same shape as input.
        return [data_shape, label_shape], [output_shape], []
    # Finally, we need to define a create_operator function
    # that will be called by the backend to create an instance of Softmax:
    def create_operator(self, ctx, shapes, dtypes):
        return BinaryEntropyLoss()
################### python customized operator end ###################



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

        init_memory_key = mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(init_memory_key, act_type="tanh"),
                                                                 axis=0),
                                              shape=(self.batch_size, self.memory_size, self.memory_key_state_dim))
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
            key_read_hidden_state = mx.sym.FullyConnected(data=slice_q_embed_data[i], num_hidden=self.q_state_dim,
                                                   weight=fnn_key_weight, bias=fnn_key_bias, name="key_read_fc")
            key_read_hidden_state = mx.sym.Activation(data=key_read_hidden_state, act_type='tanh', name="key_read_tanh")
            key_read_focus = mem.key_read(key_read_hidden_state)
            ### TODO here only compute a write weight but not write to the key

            value_write_hidden_state = mx.sym.FullyConnected(data=slice_qa_embed_data[i], num_hidden=self.qa_state_dim,
                                                   weight=fnn_value_weight, bias=fnn_value_bias, name="value_write_fc")
            value_write_hidden_state = mx.sym.Activation(data=value_write_hidden_state, act_type='tanh', name="value_write_tanh")
            value_write_focus = key_read_focus
            new_memory_value, erase_signal, add_signal, _ = mem.value_write(value_write_focus, value_write_hidden_state)

            value_read_hidden_state = mx.sym.FullyConnected(data=slice_q_embed_data[i+1], num_hidden=self.q_state_dim,
                                                          weight=fnn_key_weight, bias=fnn_key_bias, name="value_read_fc")
            value_read_hidden_state = mx.sym.Activation(data=value_read_hidden_state, act_type='tanh', name="value_read_tanh")
            value_read_focus = mem.key_read(value_read_hidden_state)
            read_value_content = mem.value_read(value_read_focus)

            ### save intermedium data
            controller_states.append(value_write_hidden_state)
            key_read_focus_l.append(key_read_focus)
            value_read_focus_l.append(value_read_focus)
            value_read_content_l.append(read_value_content)

        all_read_value_content = mx.sym.Concat(*value_read_content_l, num_args=self.seqlen, dim=0)

        pred_fc_weight = mx.sym.Variable("pred_fc_weight")
        pred_fc_bias = mx.sym.Variable("pred_fc_bias")
        pred = mx.sym.FullyConnected(data=all_read_value_content, num_hidden = self.n_question,
                                     weight=pred_fc_weight, bias=pred_fc_bias, name="final_fc")

        target = mx.sym.Reshape(data=target, shape=(-1,))

        pred_prob = mx.symbol.Custom(data=pred, label=target, name='BinaryEntropyLoss', op_type='BinaryEntropyLoss')
        # pred_prob
        return mx.sym.Group([pred_prob,
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

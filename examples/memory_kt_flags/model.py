import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from arena import Base
from arena.ops import *
from arena.utils import *
from lrua import MANN

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
        n_q = 100  # assist2015
        #n_q = 50 # synthetic
        #n_q = 436 # algebra_2005_2006_train.txt
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
                 embed_dim, control_state_dim, memory_size, memory_state_dim, k_smallest, gamma, controller,
                 num_reads, num_writes,
                 name="KT"):
        self.n_question = n_question
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.control_state_dim = control_state_dim
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.k_smallest = k_smallest
        self.gamma = gamma
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.name = name
        self.controller = controller


    def sym_gen(self):
        ### TODO input variable 'data'
        data = mx.sym.Variable('data') # (seqlen, batch_size)
        ### TODO input variable 'target'
        target = mx.sym.Variable('target') #(seqlen, batch_size)

        ### Initialize Control Network
        if self.controller == "FNN":
            fnn_weight = mx.sym.Variable("fnn_weight")
            fnn_bias = mx.sym.Variable("fnn_bias")
        elif self.controller == "LSTM":
            ### TODO input variable 'controller->layer0:init_h' / 'controller->layer0:init_c'
            init_h = [mx.sym.Variable('controller->layer0:init_h')]
            init_c = [mx.sym.Variable('controller->layer0:init_c')]
            init_h = [mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(ele, act_type="tanh"), axis=0),
                                          shape=(self.batch_size, self.control_state_dim))
                      for ele in init_h]
            init_c = [mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(ele, act_type="tanh"), axis=0),
                                          shape=(self.batch_size, self.control_state_dim))
                      for ele in init_c]
            controller = RNN(num_hidden=[self.control_state_dim], data_dim=self.embed_dim,
                             typ="lstm",
                             init_h=init_h,
                             init_c=init_c,
                             name="controller")
            controller_h = controller.init_h[0]
            controller_c = controller.init_c[0]

        ### Initialize Memory
        ### TODO input variable 'init_memory'
        init_memory = mx.sym.Variable('init_memory')
        #init_read_content = mx.sym.Variable('init_read_content')
        ### TODO input variable 'MANN->write_head:init_W_r_focus' / 'MANN->write_head:init_W_u_focus'
        init_write_W_r_focus = mx.sym.Variable('MANN->write_head:init_W_r_focus' )
        init_write_W_u_focus = mx.sym.Variable('MANN->write_head:init_W_u_focus')

        init_memory = mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(init_memory, act_type="tanh"),
                                                             axis=0),
                                          shape=(self.batch_size, self.memory_size, self.memory_state_dim))
        #init_read_content = mx.sym.broadcast_to(mx.sym.expand_dims(mx.sym.Activation(init_read_content, act_type="tanh"),
        #                                        axis=0),
        #                                        shape=(self.batch_size, self.num_reads, self.memory_state_dim))
        init_write_W_r_focus = mx.sym.SoftmaxActivation(init_write_W_r_focus)
        init_write_W_r_focus = mx.sym.broadcast_to(mx.sym.expand_dims(init_write_W_r_focus, axis=0),
                                              shape=(self.batch_size, self.num_reads, self.memory_size))
        init_write_W_u_focus = mx.sym.SoftmaxActivation(init_write_W_u_focus)
        init_write_W_u_focus = mx.sym.broadcast_to(mx.sym.expand_dims(init_write_W_u_focus, axis=0),
                                                   shape=(self.batch_size, self.num_reads, self.memory_size))
        mem = MANN(control_state_dim=self.control_state_dim,
                   memory_size=self.memory_size,
                   memory_state_dim=self.memory_state_dim,
                   k_smallest=self.k_smallest,
                   gamma=self.gamma,
                   num_reads=self.num_reads,
                   num_writes=self.num_writes,
                   init_memory=init_memory,
                   init_write_W_r_focus=init_write_W_r_focus,
                   init_write_W_u_focus=init_write_W_u_focus,
                   name="MANN")

        #read_content = init_read_content
        controller_states = []
        all_read_focus_l = []
        all_write_focus_l = []
        all_read_content_l = []
        updating_memory_l = []
        all_norm_key_l = []
        all_norm_memory_l = []
        all_similarity_score_l = []

        embed_weight = mx.sym.Variable("embed_weight")
        embed_data = mx.sym.Embedding(data=data, input_dim=self.n_question*2,
                                      weight=embed_weight, output_dim=self.embed_dim, name='embed')
        slice_embed_data = mx.sym.SliceChannel(embed_data, num_outputs=self.seqlen, axis=0, squeeze_axis=True)

        for i in range(self.seqlen):
            if self.controller == "FNN":
                q_hidden_state = mx.sym.FullyConnected(data=slice_embed_data[i], num_hidden=self.control_state_dim,
                                                       weight=fnn_weight, bias=fnn_bias, name="fc")
                q_hidden_state = mx.sym.Activation(data=q_hidden_state, act_type='tanh', name="fc_tanh")
                write_focus, erase_signal, add_signal, updated_memory = mem.write(q_hidden_state)
                read_content, read_focus, norm_key, norm_memory, similarity_score = mem.read(q_hidden_state)
                controller_states.append(q_hidden_state)
            elif self.controller == "LSTM":
                controller_h, controller_c = \
                    controller.step(data=slice_embed_data[i],
                                    prev_h=controller_h, prev_c=controller_c,
                                    seq_len=1,
                                    ret_typ="state")
                controller_h = controller_h[0]
                controller_c = controller_c[0]
                write_focus, erase_signal, add_signal, updated_memory = mem.write(controller_h)
                read_content, read_focus, norm_key, norm_memory, similarity_score = mem.read(controller_h)
                controller_states.append(controller_h)

            # internal results used for checking
            all_norm_key_l.append(norm_key)
            all_norm_memory_l.append(norm_memory)
            all_similarity_score_l.append(similarity_score)
            ### save intermedium data
            all_read_focus_l.append(read_focus)
            all_write_focus_l.append(write_focus)
            all_read_content_l.append(read_content)

        all_read_content = mx.sym.Concat(*all_read_content_l, num_args=self.seqlen, dim=0)

        fc_weight = mx.sym.Variable("fc_weight")
        fc_bias = mx.sym.Variable("fc_bias")
        new_key = mx.sym.Activation(data=all_read_content, act_type='tanh', name=self.name + ":return_key")
        pred = mx.sym.FullyConnected(data=new_key, num_hidden = self.n_question,
                                     weight=fc_weight, bias=fc_bias, name="final_fc")
        target = mx.sym.Reshape(data=target, shape=(-1,))
        pred_prob = mx.symbol.Custom(data=pred, label=target, name='BinaryEntropyLoss', op_type='BinaryEntropyLoss')

        return mx.sym.Group([pred_prob,
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*controller_states, dim=0,
                                               num_args=len(controller_states)),
                                 shape=(self.seqlen, -1, self.control_state_dim))),
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*all_norm_key_l, dim=0,
                                               num_args=len(all_norm_key_l)),
                                 shape=(self.seqlen, -1, self.memory_state_dim))),
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*all_norm_memory_l, dim=0,
                                               num_args=len(all_norm_memory_l)),
                                 shape=(self.seqlen, -1, self.memory_size))),
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*all_similarity_score_l, dim=0,
                                               num_args=len(all_similarity_score_l)),
                                 shape=(self.seqlen, -1, self.memory_size))),
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*all_read_content_l, dim=0,
                                               num_args=len(all_read_content_l)),
                                 shape=(self.seqlen, -1, self.memory_state_dim))),
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*all_read_focus_l, dim=0,
                                               num_args=len(all_read_focus_l)),
                                 shape=(self.seqlen, -1, self.memory_size))),
                             mx.sym.BlockGrad(mx.sym.Reshape(
                                 mx.sym.Concat(*all_write_focus_l, dim=0,
                                               num_args=len(all_write_focus_l)),
                                 shape=(self.seqlen, -1, self.memory_size)))
                             ])

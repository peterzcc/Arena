import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from arena import Base
from arena.ops import LSTM
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
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        #print "x", x
        y = 1.0/(1.0+np.exp(-x))
        #print "y", y
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        n_q = 111
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
    def __init__(self, n_question, seqlen,
                 embed_dim, control_state_dim, memory_size, memory_state_dim, k_smallest,
                 num_reads, num_writes,
                 name="KT"):
        self.n_question = n_question
        self.seqlen = seqlen
        self.embed_dim = embed_dim
        self.control_state_dim = control_state_dim
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.k_smallest = k_smallest
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.name = name

    def sym_gen(self):
        # TODO input variable 'data'
        data = mx.sym.Variable('data') # (batch_size, seqlen)
        # TODO input variable 'target'
        target = mx.sym.Variable('target') #(batch_size, seqlen)

        ### Initialize Control Network
        controller = LSTM(num_hidden=self.control_state_dim, name="controller")
        controller_h = controller.init_h[0]
        controller_c = controller.init_c[0]
        ### Initialize Memory
        # TODO input variable 'init_memory'
        init_memory = mx.sym.Variable('init_memory')
        # TODO input variable 'init_read_W_r_focus' / 'init_read_W_w_focus' / 'init_read_W_u_focus'
        init_read_W_r_focus = [mx.sym.Variable('MANN->read_head%d:init_W_r_focus' % i) for i in range(self.num_reads)]
        init_read_W_w_focus = [mx.sym.Variable('MANN->read_head%d:init_W_w_focus' % i) for i in range(self.num_reads)]
        init_read_W_u_focus = [mx.sym.Variable('MANN->read_head%d:init_W_u_focus' % i) for i in range(self.num_reads)]
        # TODO input variable 'init_write_W_r_focus' / 'init_write_W_w_focus' / 'init_write_W_u_focus'
        init_write_W_r_focus = [mx.sym.Variable('MANN->write_head%d:init_W_r_focus' % i) for i in range(self.num_writes)]
        init_write_W_w_focus = [mx.sym.Variable('MANN->write_head%d:init_W_w_focus' % i) for i in range(self.num_writes)]
        init_write_W_u_focus = [mx.sym.Variable('MANN->write_head%d:init_W_u_focus' % i) for i in range(self.num_writes)]

        ### def __init__(self, control_state_dim, memory_size, memory_state_dim, k_smallest,
        ###         num_reads, num_writes, init_memory=None,
        ###         init_read_W_r_focus=None, init_read_W_w_focus=None, init_read_W_u_focus=None,
        ###         init_write_W_r_focus=None, init_write_W_w_focus=None, init_write_W_u_focus=None,
        ###         name="MANN"):
        mem = MANN(control_state_dim=self.control_state_dim,
                   memory_size=self.memory_size,
                   memory_state_dim=self.memory_state_dim,
                   k_smallest=self.k_smallest,
                   num_reads=self.num_reads,
                   num_writes=self.num_writes,
                   init_memory=init_memory,
                   init_read_W_r_focus=init_read_W_r_focus,
                   init_read_W_w_focus=init_read_W_w_focus,
                   init_read_W_u_focus=init_read_W_u_focus,
                   init_write_W_r_focus=init_write_W_r_focus,
                   init_write_W_w_focus=init_write_W_w_focus,
                   init_write_W_u_focus=init_write_W_u_focus,
                   name="MANN") # TODO init_read_focus / init_write_focus
        controller_states = []
        all_read_focus_l = []
        all_write_focus_l = []
        all_read_content_l = []
        ### model start
        ### Step1:
        ###     input_data      -(embed)->          embed_data
        ### (batch_size,seqlen) -(embed)-> (batch_size, seqlen, embed_dim)
        embed_weight = mx.sym.Variable("embed_weight")
        embed_data = mx.sym.Embedding(data=data, input_dim=self.n_question*2,
                                      weight=embed_weight, output_dim=self.embed_dim, name='embed')
        ### Step2:
        ### at each time step:
        ###      one_time_input     -(LSTM)->       controller_h
        ### (batch_size, embed_dim) -(LSTM)-> (batch_size, control_state_dim)
        for i in range(self.seqlen):
            controller_h, controller_c = controller.step(data=embed_data[:,i,:],
                                                         prev_h=controller_h, prev_c=controller_c,
                                                         seq_length=1)
            controller_h = controller_h[0]
            controller_c = controller_c[0]
            ### Step3:
            ###         controller_h            -(read)->       read_content_l          /       read_focus_l
            ### (batch_size, control_state_dim) -(read)-> (batch_size, memory_state_dim)/(batch_size, memory_size)
            read_content_l, read_focus_l = mem.read(controller_h)
            ### Step4:
            ###         controller_h            -(write)->        write_focus_l
            ### (batch_size, control_state_dim) -(write)-> (batch_size, memory_size)
            write_focus_l = mem.write(controller_h)
            ### save intermedium data
            controller_states.append(controller_h)
            all_read_focus_l.append(read_focus_l[0])
            all_write_focus_l.append(write_focus_l[0])
            all_read_content_l.append(read_content_l) # TODO check read_content format
        ### Step5:
        ###       all_read_content_l        -(Concat)->        all_read_content
        ### [(batch_size, memory_state_dim)] -(Concat)-> (batch_size, memory_size) TODO need to check
        all_read_content = mx.sym.Concat(*all_read_content_l, dim=0)
        ### Step6:
        ###         all_read_content            -(FC)->        write_focus_l
        ### (batch_size, control_state_dim) -(FC)-> (batch_size, memory_size)
        fc_weight = mx.sym.Variable("fc_weight")
        fc_bias = mx.sym.Variable("fc_bias")
        pred = mx.sym.FullyConnected(data=all_read_content, num_hidden = self.n_question,
                                     weight=fc_weight, bias=fc_bias, name="final_fc")
        # TODO label format needs to check
        target = mx.sym.transpose(data=target)
        target = mx.sym.Reshape(data=target, shape=(-1,))
        be = mx.symbol.Custom(data=pred, label=target, name='BinaryEntropyLoss', op_type='BinaryEntropyLoss')
        return be



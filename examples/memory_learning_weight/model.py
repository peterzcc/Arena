import numpy as np
from arena.ops import LSTM
from mem import WLMN
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
        n_q = 111 # assist2009
        #n_q = 50 # synthetic
        #n_q = 1084 # algebra_2005_2006_train.txt
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
                 q_embed_dim, q_state_dim, qa_embed_dim, qa_state_dim,
                 memory_size, memory_state_dim, name="KT"):
        self.n_question = n_question
        self.seqlen = seqlen
        self.q_embed_dim = q_embed_dim
        self.q_state_dim = q_state_dim
        self.qa_embed_dim = qa_embed_dim
        self.qa_state_dim = qa_state_dim
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.name = name

    def sym_gen(self):
        ### TODO input variable 'q_data'
        q_data = mx.sym.Variable('q_data') # (seqlen, batch_size)
        ### TODO input variable 'qa_data'
        qa_data = mx.sym.Variable('qa_data')  # (seqlen, batch_size)
        ### TODO input variable 'target'
        target = mx.sym.Variable('target') #(seqlen, batch_size)

        ### Initialize Control Networks
        fnn_weight = mx.sym.Variable("fnn_weight")
        fnn_bias = mx.sym.Variable("fnn_bias")
        read_weight = mx.sym.Variable("p_read_weight")
        read_bias = mx.sym.Variable("p_read_bias")
        write_weight = mx.sym.Variable("p_write_weight")
        write_bias = mx.sym.Variable("p_write_bias")

        controller = LSTM(num_hidden=self.qa_state_dim, name="controller")
        ### TODO input variable 'controller_init_h'/'controller_init_c'
        controller_h = controller.init_h[0]
        controller_c = controller.init_c[0]
        ### Initialize Memory
        ### TODO input variable 'init_memory'
        init_memory = mx.sym.Variable('init_memory')

        mem = WLMN(memory_size=self.memory_size,
                   memory_state_dim=self.memory_state_dim,
                   init_memory=init_memory,
                   name="WLMN")
        ### model start
        q_embed_weight = mx.sym.Variable("q_embed_weight")
        q_embed_data = mx.sym.Embedding(data=q_data, input_dim=self.n_question,
                                        weight=q_embed_weight, output_dim=self.q_embed_dim, name='q_embed')
        slice_q_embed_data = mx.sym.SliceChannel(q_embed_data, num_outputs=self.seqlen, axis=0, squeeze_axis=True)

        qa_embed_weight = mx.sym.Variable("qa_embed_weight")
        qa_embed_data = mx.sym.Embedding(data=qa_data, input_dim=self.n_question*2,
                                         weight=qa_embed_weight, output_dim=self.qa_embed_dim, name='qa_embed')
        slice_qa_embed_data = mx.sym.SliceChannel(qa_embed_data, num_outputs=self.seqlen, axis=0, squeeze_axis=True)

        controller_states = []
        all_read_focus_l = []
        all_write_focus_l = []
        all_read_content_l = []
        for i in range(self.seqlen):
            q_hidden_state = mx.sym.FullyConnected(data=slice_q_embed_data[i], num_hidden=self.q_state_dim, weight=fnn_weight, bias=fnn_bias, name="q_fc")
            q_hidden_state = mx.sym.Activation(data=q_hidden_state, act_type='tanh', name="q_fc_tanh")

            read_focus = mx.sym.FullyConnected(data=q_hidden_state, num_hidden=self.memory_size, weight=read_weight, bias=read_bias, name="q_read")
            read_focus = mx.sym.Activation(data=read_focus, act_type='tanh', name="q_read_tanh")
            read_focus = mx.sym.Reshape(read_focus, shape=(-1, 1, self.memory_size))

            write_focus = mx.sym.FullyConnected(data=q_hidden_state, num_hidden=self.memory_size,weight=write_weight, bias=write_bias, name="q_write")
            write_focus = mx.sym.Activation(data=write_focus, act_type='tanh', name="q_write_tanh")
            write_focus = mx.sym.Reshape(write_focus, shape=(-1, 1, self.memory_size))

            ###  LSTM for qa_sequence
            controller_h, controller_c = controller.step(data=slice_qa_embed_data[i],
                                                         prev_h=controller_h, prev_c=controller_c,
                                                         seq_length=1)
            controller_h = controller_h[0]
            controller_c = controller_c[0]

            new_memory = mem.write(write_focus, controller_h)
            read_content  = mem.read(read_focus, controller_h)

            controller_states.append(controller_h)
            all_read_focus_l.append(read_focus)
            all_write_focus_l.append(write_focus)
            all_read_content_l.append(read_content)

        all_read_content = mx.sym.Concat(*all_read_content_l, num_args=self.seqlen, dim=0)

        pred_fc_weight = mx.sym.Variable("pred_fc_weight")
        pred_fc_bias = mx.sym.Variable("pred_fc_bias")
        pred = mx.sym.FullyConnected(data=all_read_content, num_hidden = self.n_question,
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

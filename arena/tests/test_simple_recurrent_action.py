import mxnet as mx
from collections import namedtuple
import arena
from arena.utils import *
from arena import Base
import logging
import numpy
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

'''
Name: LogNormalPolicy
Usage: This OP outputs actions generated by a policy with normal distribution.
       The loss function for backward operation is set to be \sum_i - log(N(a_i|m_i, v_i)) * R_i
'''
class LogNormalPolicy(mx.operator.NumpyOp):
    def __init__(self, rng=get_numpy_rng()):
        super(LogNormalPolicy, self).__init__(need_top_grad=False)
        self.rng = rng

    def list_arguments(self):
        return ['mean', 'var', 'score']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        mean_shape = in_shape[0]
        var_shape = in_shape[1]
        score_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [mean_shape, var_shape, score_shape], \
                   [output_shape]

    def forward(self, in_data, out_data):
        mean = in_data[0]
        var = in_data[1]
        if 1 == var.ndim:
            out_data[0][:] = numpy.sqrt(var.reshape((var.shape[0], 1))) \
                             * self.rng.randn(*mean.shape) + mean
        else:
            out_data[0][:] = numpy.sqrt(var) * self.rng.randn(*mean.shape) + mean

    def backward(self, out_grad, in_data, out_data, in_grad):
        mean = in_data[0]
        var = in_data[1]
        action = out_data[0]
        score = in_data[2]
        if 1 == var.ndim :
            grad_mu = in_grad[0]
            grad_mu[:] = - (action - mean) * score.reshape((score.shape[0], 1)) / \
                         var.reshape((var.shape[0], 1))
        else:
            grad_mu = in_grad[0]
            grad_var = in_grad[1]
            grad_mu[:] = - (action - mean) * score.reshape((score.shape[0], 1)) / var
            grad_var[:] = - numpy.square(action - mean) * score.reshape((score.shape[0], 1))\
                          / numpy.square(var) / 2

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="lstm%d_t%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="lstm%d_t%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="lstm%d_t%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def build_recurrent_sym(seq_len, num_lstm_layer=1):
    param_cells = []
    last_states = []
    mean_fc_weight = mx.sym.Variable("mean_fc_weight")
    mean_fc_bias = mx.sym.Variable("mean_fc_bias")
    var_fc_weight = mx.sym.Variable("var_fc_weight")
    var_fc_bias = mx.sym.Variable("var_fc_bias")
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("lstm%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("lstm%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("lstm%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("lstm%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("lstm%d_init_c" % i),
                          h=mx.sym.Variable("lstm%d_init_h" % i))
        last_states.append(state)
    assert (len(last_states) == num_lstm_layer)
    data = mx.sym.Variable("data")
    outputs = []
    means = []
    vars = []
    for seqidx in range(seq_len):
        if seqidx == 0:
            hidden = data
        else:
            hidden = outputs[-1]
        # Stacking LSTM Layers
        for i in range(num_lstm_layer):
            next_state = lstm(50, indata=hidden, prev_state=last_states[i], param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=0.)
            hidden = next_state.h
            last_states[i] = next_state
        mean = mx.sym.FullyConnected(data=hidden, weight=mean_fc_weight, bias=mean_fc_bias,
                                     num_hidden=4, name="mean_fc_t%d" % seqidx)
        means.append(mx.sym.BlockGrad(data=mean))
        var = mx.sym.FullyConnected(data=hidden, weight=var_fc_weight, bias=var_fc_bias,
                                     num_hidden=4, name="var_fc_t%d" % seqidx)
        var = mx.sym.Activation(data=var, act_type="softrelu", name="var_fc_softplus_t%d" % seqidx)
        vars.append(mx.sym.BlockGrad(data=var))
        LogNormalPolicyOp = LogNormalPolicy()
        out = LogNormalPolicyOp(mean=mean, var=var, name="policy_t%d" %seqidx)
        outputs.append(out)
    net = mx.sym.Group(outputs + means + vars)
    return net

def update_line(hl, fig, ax, new_x, new_y):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_x))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_y))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

def simple_addition_game(data, action):
    return 10 * (numpy.square(action - sum(data.values())/len(data.values())).mean(axis=1) < 1) + \
           10 * (numpy.square(action - sum(data.values())/len(data.values())).mean(axis=1) < 0.5) + \
           10 * (numpy.square(action - sum(data.values()) / len(data.values())).mean(axis=1) < 0.2) + \
           10 * (numpy.square(action - sum(data.values()) / len(data.values())).mean(axis=1) < 0.05) + 1


def simple_sequence_generation_game(data, actions):
    total_time_steps = len(actions)
    ground_truth = [data.values()[0] * (2*(t%3)/(t+1)) for t in range(total_time_steps)]
    scores = [0]*total_time_steps
    for i, (truth, action) in enumerate(zip(ground_truth, actions)):
        scores[i] = 1 * (numpy.square(action - truth).mean(axis=1) < 0.001) + \
                    1 * (numpy.square(action - truth).mean(axis=1) < 0.0005) + \
                    1 * (numpy.square(action - truth).mean(axis=1) < 0.0002) + \
                    1 * (numpy.square(action - truth).mean(axis=1) < 0.0001) + \
                    1 * (numpy.square(action - truth).mean(axis=1) < 0.00005) + \
                    + 1
    return scores

time_step = 20
minibatch_size = 20
layer_number = 2
net = build_recurrent_sym(time_step, layer_number)
print net.list_arguments()
data_shapes = dict([("data", (minibatch_size, 4))] +
                   [("lstm%d_init_c" % i, (minibatch_size, 50)) for i in range(layer_number)] +
                   [("lstm%d_init_h" % i, (minibatch_size, 50)) for i in range(layer_number)] +
                   [('policy_t%d_score' % i, (minibatch_size, )) for i in range(time_step)])
print data_shapes
qnet = Base(data_shapes=data_shapes, sym=net, name='PolicyNet',
            initializer=mx.initializer.Xavier(factor_type="in", magnitude=1.0),
            ctx=mx.gpu())

optimizer = mx.optimizer.create(name='sgd', learning_rate=0.00001,
                                clip_gradient=None,
                                rescale_grad=1.0/minibatch_size, wd=0.00001)
updater = mx.optimizer.get_updater(optimizer)
qnet.print_stat()
baseline = numpy.zeros((time_step,))
decay_factor = 0.5
for epoch in range(10000):
    data = [("data", numpy.random.rand(minibatch_size, 4))]
    data_ndarray = {k: nd.array(v, ctx=mx.gpu()) for k, v in data}
    outputs = qnet.forward(batch_size=minibatch_size, **data_ndarray)
    actions = get_npy_list(outputs[:time_step])
    means = get_npy_list(outputs[time_step:(time_step*2)])
    vars = get_npy_list(outputs[(time_step*2):(time_step*3)])
    scores = simple_sequence_generation_game(dict(data), actions)
    scores = [score * pow(decay_factor, time_step - 1 - i) for i, score in enumerate(scores)]
    q_estimation = numpy.cumsum(scores[::-1], axis=0)[::-1]
    baseline = baseline - 0.01 * (baseline - q_estimation.mean(axis=1))
    qnet.backward(batch_size=minibatch_size,
                  **dict(data_ndarray.items() + [("policy_t%d_score" %(i), score)
                                                 for i, score in enumerate(q_estimation
                                                            - baseline.reshape(time_step, 1))]))
    qnet.update(updater)
    print 'scores=', numpy.mean(scores), 'mean_score=', numpy.mean(simple_sequence_generation_game(dict(data), means)), \
          'baseline=', baseline

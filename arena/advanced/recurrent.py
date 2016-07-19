import mxnet as mx
from collections import namedtuple

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMLayerProp = namedtuple("LSTMLayerProp", ['num_hidden', 'dropout'])

def step_lstm(num_hidden, indata, prev_state, param, dropout=0., layeridx=0, prefix='',
              postfix=''):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="%slstm%d_i2h%s" % (prefix, layeridx, postfix))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="%slstm%d_h2h%s" % (prefix, layeridx, postfix))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="%slstm%d_slice%s" %
                                           (prefix, layeridx, postfix))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


#TODO Revise the naming strategy
def step_stack_lstm(indata, prev_states, lstm_props, params, prefix='', postfix=''):
    assert (len(lstm_props) == len(params)) and (len(prev_states) == len(lstm_props))
    new_states = []
    lstm_input = indata
    for i, lstm_prop in enumerate(lstm_props):
        new_states.append(step_lstm(num_hidden=lstm_prop.num_hidden, indata=lstm_input,
                                    param=params[i], prev_state=prev_states[i], layeridx=i,
                                    prefix=prefix, postfix=postfix))
        lstm_input = new_states[-1].h
    return new_states
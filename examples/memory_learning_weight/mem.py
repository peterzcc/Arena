from arena.utils import *
import mxnet as mx

class WLMN(object):
    def __init__(self, memory_size, memory_state_dim, init_memory=None, name="WLMN"):
        self.name = name
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim

        self.key_weight = mx.sym.Variable(name=name + ":key_weight")
        self.key_bias = mx.sym.Variable(name=name + ":key_bias")

        self.init_memory = mx.sym.Variable(self.name + ":init_memory") if init_memory is None\
                                                                               else init_memory
        self.read_counter = 0
        self.write_counter = 0
        self.memory = self.init_memory


    def read(self, read_focus, control_input):
        assert isinstance(control_input, mx.symbol.Symbol)
        value_content = mx.sym.batch_dot(read_focus, self.memory)
        value_content = mx.sym.sum(value_content, axis=1)  # Shape (batch_size, memory_value_state_dim)
        self.read_counter += 1
        return value_content

    def write(self, write_focus, control_input):
        assert isinstance(control_input, mx.symbol.Symbol)
        key = mx.sym.FullyConnected(data=control_input,
                                    num_hidden=self.memory_state_dim,
                                    weight=self.key_weight,
                                    bias=self.key_bias)  # Shape: (batch_size, num_writes*memory_value_state_dim)
        key = mx.sym.Reshape(key, shape=(-1, 1, self.memory_state_dim))
        key = mx.sym.Activation(data=key, act_type='tanh', name=self.name + ":key")

        self.memory = self.memory + mx.sym.batch_dot(mx.sym.SwapAxis(write_focus, dim1=1, dim2=2), key)
        self.write_counter += 1
        return self.memory

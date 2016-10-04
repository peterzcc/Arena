from arena.operators import *
from arena.utils import *
import mxnet as mx

class KVMNHeadGroup(object):
    def __init__(self, memory_size, memory_state_dim, is_write, num_heads=1, name="KVMNHeadGroup"):
        """
        :param control_state_dim:
        :param memory_size:
        :param memory_state_dim:
        :param k_smallest:
        :param gamma:
        :param num_heads:
        :param init_W_r_focus: Shape (batch_size, num_heads, memory_size)
        :param init_W_u_focus: Shape (batch_size, num_heads, memory_size)
        :param name:
        """
        self.name = name
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.num_heads = num_heads


        self.is_write = is_write
        if self.is_write:
            self.erase_signal_weight = mx.sym.Variable(name=name + ":erase_signal_weight")
            self.erase_signal_bias = mx.sym.Variable(name=name + ":erase_signal_bias")
            self.add_signal_weight = mx.sym.Variable(name=name + ":add_signal_weight")
            self.add_signal_bias = mx.sym.Variable(name=name + ":add_signal_bias")
        else:
            self.key_weight = mx.sym.Variable(name=name + ":key_weight")
            self.key_bias = mx.sym.Variable(name=name + ":key_bias")
            self.beta_weight = mx.sym.Variable(name=name + ":beta_weight")
            self.beta_bias = mx.sym.Variable(name=name + ":beta_bias")

        self.address_counter = 0


    def addressing(self, control_input, memory):
        """
            :param  control_input: Shape (batch_size, control_state_dim)
            :param  memory:        Shape (batch_size, memory_size, memory_state_dim)
            :return W_r:           Shape (batch_size, num_heads, memory_size)
            :return W_w:           Shape (batch_size, num_heads, memory_size)
        """
        # key
        key = mx.sym.FullyConnected(data=control_input,
                                    num_hidden=self.num_heads * self.memory_state_dim,
                                    weight=self.key_weight,
                                    bias=self.key_bias)  # Shape: (batch_size, num_reads*memory_state_dim)
        key = mx.sym.Reshape(key, shape=(-1, self.num_heads, self.memory_state_dim))  # Shape: (batch_size, num_heads, memory_state_dim)
        key = mx.sym.Activation(data=key, act_type='tanh', name=self.name + ":key")
        # beta
        beta = mx.sym.FullyConnected(data=control_input,
                                     num_hidden=self.num_heads,
                                     weight=self.beta_weight,
                                     bias=self.beta_bias)
        beta = mx.sym.expand_dims(beta, axis=2)  # Shape: (batch_size, num_heads, 1)
        beta = mx.sym.Activation(data=beta, act_type='softrelu', name=self.name + ":beta")

        # compute cosine similarity
        norm_key = ArenaSym.normalize_channel(key, axis=2)
        norm_memory = ArenaSym.normalize_channel(memory, axis=2)
        ### key            (batch_size, num_heads, memory_state_dim)
        ### swapped_memory (batch_size, memory_state_dim, memory_size)
        similarity_score = mx.sym.batch_dot(norm_key, mx.sym.SwapAxis(norm_memory, dim1=1, dim2=2)) # Shape: (batch_size, num_heads, memory_size)
        # compute read weight
        W_r = mx.sym.Reshape(mx.sym.SoftmaxActivation(mx.sym.Reshape(mx.sym.broadcast_mul(beta, similarity_score),
                                                                    shape=(-1, self.memory_size))),
                            shape=(-1, self.num_heads, self.memory_size))  # Shape: (batch_size, num_heads, memory_size)
        return  W_r

    def read(self, memory, control_input=None, read_focus=None ):
        """Read from the memory
        Parameters
            control_input: Shape (batch_size, control_state_dim)
            memory: Shape (batch_size, memory_size, memory_state_dim)
        ------------------------------------------------------
        Returns
            The read content --> Shape (batch_size,  memory_state_dim)
            The read focus   --> Shape (batch_size, num_heads, memory_size)
        """
        if read_focus is None:
            read_focus = self.addressing(control_input=control_input, memory=memory)
        content = mx.sym.batch_dot(read_focus, memory)
        content = mx.sym.sum(content, axis=1) # Shape (batch_size, memory_state_dim)
        return content, read_focus


    def write(self, control_input, memory, write_focus=None):
        """Write into the memory
        Parameters
            control_input: Shape (batch_size, control_state_dim)
            memory: Shape (batch_size, memory_size, memory_state_dim)
        ------------------------------------------------------
        Returns
            new_memory      --> Shape (batch_size, memory_size, memory_state_dim)
            The write focus --> Shape (batch_size, num_heads, memory_size)
        """
        assert self.is_write
        if write_focus is None:
            write_focus = self.addressing(control_input=control_input, memory=memory)  # Shape Shape (batch_size, num_heads, memory_size)

        erase_signal = mx.sym.FullyConnected(data=control_input,
                                             num_hidden=self.num_heads * self.memory_state_dim,
                                             weight=self.erase_signal_weight,
                                             bias=self.erase_signal_bias)
        erase_signal = mx.sym.Reshape(erase_signal, shape=(0, self.num_heads, self.memory_state_dim))
        erase_signal = mx.sym.Activation(data=erase_signal, act_type='sigmoid', name=self.name + "_erase_signal")
        ### key = add_signal
        add_signal = mx.sym.FullyConnected(data=control_input,
                                           num_hidden=self.num_heads * self.memory_state_dim,
                                           weight=self.add_signal_weight,
                                           bias=self.add_signal_bias)
        add_signal = mx.sym.Reshape(add_signal, shape=(0, self.num_heads, self.memory_state_dim))
        add_signal = mx.sym.Activation(data=add_signal, act_type='tanh', name=self.name + "_add_signal")
        erase_mult = 1 - mx.sym.batch_dot(mx.sym.Reshape(write_focus, shape=(-1, self.memory_size, 1)),
                                          mx.sym.Reshape(erase_signal, shape=(-1, 1, self.memory_state_dim)))
        if self.num_heads > 1:
            erase_mult = mx.sym.Reshape(erase_mult, shape=(-1, self.num_heads, self.memory_size, self.memory_state_dim))
            erase_mult = mx.sym.exp(mx.sym.sum(mx.sym.log(erase_mult), axis=1))
        aggre_add_signal = mx.sym.batch_dot(mx.sym.SwapAxis(write_focus, dim1=1, dim2=2), add_signal)
        new_memory = memory * erase_mult + aggre_add_signal
        ### previously without erase
        ### new_memory = memory + mx.sym.batch_dot( mx.sym.SwapAxis(write_focus, dim1=1, dim2=2),key)
        ### return new_memory, write_focus
        return new_memory, erase_signal, add_signal, write_focus


class KVMN(object):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim,
                 num_heads, init_memory_key=None, init_memory_value=None,
                 name="KVMN"):
        """
        :param control_state_dim:
        :param memory_size:
        :param memory_state_dim:
        :param k_smallest:
        :param gamma:
        :param num_reads:
        :param num_writes:
        :param init_memory: Shape (batch_size, memory_size, memory_state_dim)
        :param init_write_W_r_focus: Shape (batch_size, num_write, memory_size)
        :param init_write_W_u_focus: Shape (batch_size, num_write, memory_size)
        :param name:
        """
        self.name = name
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.num_heads = num_heads

        self.init_memory_key = mx.sym.Variable(self.name + ":init_memory_key") if init_memory_key is None\
                                                                               else init_memory_key
        self.init_memory_value = mx.sym.Variable(self.name + ":init_memory_value") if init_memory_value is None\
                                                                               else init_memory_value

        self.key_head = KVMNHeadGroup(memory_size = self.memory_size,
                                      memory_state_dim = self.memory_key_state_dim,
                                      is_write = False,
                                      num_heads = self.num_heads,
                                      name = self.name + "->key_head")
        self.value_head = KVMNHeadGroup(memory_size=self.memory_size,
                                        memory_state_dim=self.memory_value_state_dim,
                                        is_write=True,
                                        num_heads=self.num_heads,
                                        name=self.name + "->value_head")

        self.memory_key = self.init_memory_key
        self.memory_value = self.init_memory_value


    def key_read(self, control_input):
        assert isinstance(control_input, mx.symbol.Symbol)
        read_focus = self.key_head.addressing(control_input=control_input, memory=self.memory_key)
        return read_focus



    def value_write(self, write_focus, control_input):
        assert isinstance(control_input, mx.symbol.Symbol)
        self.memory_value, erase_signal, add_signal, write_focus = self.value_head.write(control_input=control_input, memory=self.memory_value, write_focus=write_focus)
        return self.memory_value,erase_signal, add_signal, write_focus

    def value_read(self, read_focus):
        read_content, read_focus = self.value_head.read(memory=self.memory_value, read_focus=read_focus)
        return read_content




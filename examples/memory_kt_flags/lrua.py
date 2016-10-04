from arena.operators import *
from arena.utils import *
import mxnet as mx

class MANNHeadGroup(object):
    def __init__(self, control_state_dim, memory_size, memory_state_dim, k_smallest, gamma,
                 num_heads=1,
                 init_W_r_focus=None, init_W_u_focus=None,
                 is_write=False, name="MANNHeadGroup"):
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
        self.control_state_dim = control_state_dim
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.k_smallest = k_smallest
        self.num_heads = num_heads
        self.gamma = gamma

        self.key_weight = mx.sym.Variable(name=name + ":key_weight")
        self.key_bias = mx.sym.Variable(name=name + ":key_bias")
        self.beta_weight = mx.sym.Variable(name=name + ":beta_weight")
        self.beta_bias = mx.sym.Variable(name=name + ":beta_bias")

        self.is_write = is_write

        if self.is_write:
            self.alpha_weight = mx.sym.Variable(name=name + ":alpha_weight")
            self.alpha_bias = mx.sym.Variable(name=name + ":alpha_bias")
            self.last_W_r_focus = init_W_r_focus if init_W_r_focus is not None \
                else mx.sym.Variable(name=name + ":init_W_r_focus")
            self.last_W_u_focus = init_W_u_focus if init_W_u_focus is not None \
                else mx.sym.Variable(name=name + ":init_W_u_focus")
            self.erase_signal_weight = mx.sym.Variable(name=name + ":erase_signal_weight")
            self.erase_signal_bias = mx.sym.Variable(name=name + ":erase_signal_bias")
            self.add_signal_weight = mx.sym.Variable(name=name + ":add_signal_weight")
            self.add_signal_bias = mx.sym.Variable(name=name + ":add_signal_bias")

        self.address_counter = 0

    # Controll is FNN

    def addressing(self, control_input, memory):
        """
            :param  control_input: Shape (batch_size, control_state_dim)
            :param  memory:        Shape (batch_size, memory_size, memory_state_dim)
            :return W_r:           Shape (batch_size, num_heads, memory_size)
            :return W_w:           Shape (batch_size, num_heads, memory_size)
        """
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
        W_r = mx.sym.Reshape(mx.sym.SoftmaxActivation(
                            mx.sym.Reshape(mx.sym.broadcast_mul(beta, similarity_score),shape=(-1, self.memory_size))),
                            shape=(-1, self.num_heads, self.memory_size))  # Shape: (batch_size, num_heads, memory_size)
        if self.is_write == False:
            return norm_key, norm_memory, similarity_score, W_r
        else:
            # Alpha
            alpha = mx.sym.FullyConnected(data=control_input,
                                          num_hidden=self.num_heads,
                                          weight=self.alpha_weight,
                                          bias=self.alpha_bias)
            alpha = mx.sym.expand_dims(alpha, axis=2)  # Shape: (batch_size, num_heads, 1)
            alpha = mx.sym.Activation(data=alpha, act_type='sigmoid', name=self.name + ":alpha")
            # compute write weight
            #last_W_lu = self.last_W_u_focus
            last_W_lu = mx.sym.Reshape(mx.sym.k_smallest_flags(mx.sym.Reshape(self.last_W_u_focus,
                                                                              shape=(-1, self.memory_size)),
                                                               k=self.k_smallest),
                                       shape=(-1, self.num_heads, self.memory_size))# Shape (batch_size, num_heads, memory_size)
            #last_W_r_focus_flag = 1.0 - mx.sym.Reshape(mx.sym.k_smallest_flags(mx.sym.Reshape(self.last_W_r_focus, shape = (-1,self.memory_size)),
            #                                                             k=self.k_largest),
            #                                     shape=(-1, self.num_heads, self.memory_size))

            W_w = mx.sym.broadcast_mul(alpha, self.last_W_r_focus) + \
                  mx.sym.broadcast_mul((1.0 - alpha), last_W_lu)
            W_u = self.gamma * self.last_W_u_focus + W_r + W_w  # Shape (batch_size, memory_size)
            self.last_W_r_focus = W_r
            self.last_W_u_focus = W_u
            return  W_w

    def read(self, control_input, memory):
        """Read from the memory
        Parameters
            control_input: Shape (batch_size, control_state_dim)
            memory: Shape (batch_size, memory_size, memory_state_dim)
        ------------------------------------------------------
        Returns
            The read content --> Shape (batch_size,  memory_state_dim)
            The read focus   --> Shape (batch_size, num_heads, memory_size)
        """
        assert not self.is_write
        norm_key, norm_memory, similarity_score, read_focus = self.addressing(control_input=control_input, memory=memory)
        content = mx.sym.batch_dot(read_focus, memory)
        content = mx.sym.sum(content, axis=1) # Shape (batch_size, memory_state_dim)
        return content, read_focus, norm_key, norm_memory, similarity_score

    def write(self, control_input, memory):
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
        write_focus = self.addressing(control_input=control_input, memory=memory) # Shape Shape (batch_size, num_heads, memory_size)
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
            erase_mult = mx.sym.Reshape(erase_mult,shape=(-1, self.num_heads,self.memory_size, self.memory_state_dim))
            erase_mult = mx.sym.exp(mx.sym.sum(mx.sym.log(erase_mult), axis=1))
        aggre_add_signal = mx.sym.batch_dot(mx.sym.SwapAxis(write_focus, dim1=1, dim2=2), add_signal)
        new_memory = memory * erase_mult + aggre_add_signal
        ### previously without erase
        ### new_memory = memory + mx.sym.batch_dot( mx.sym.SwapAxis(write_focus, dim1=1, dim2=2),key)
        ### return new_memory, write_focus
        return new_memory, erase_signal, add_signal, write_focus


class MANN(object):
    def __init__(self, control_state_dim, memory_size, memory_state_dim, k_smallest, gamma,
                 num_reads, num_writes, init_memory=None,
                 init_read_W_r_focus=None, #init_read_W_w_focus=None, init_read_W_u_focus=None,
                 init_write_W_r_focus=None, init_write_W_w_focus=None, init_write_W_u_focus=None,
                 name="MANN"):
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
        self.control_state_dim = control_state_dim
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.k_smallest = k_smallest
        self.gamma = gamma

        self.init_memory = mx.sym.Variable(self.name + ":init_memory") if init_memory is None\
                                                                       else init_memory
        self.init_write_W_r_focus = mx.sym.Variable(self.name + ":init_write_W_r_focus") if init_write_W_r_focus is None\
                                                                                       else init_write_W_r_focus
        self.init_write_W_u_focus = mx.sym.Variable(self.name + ":init_write_W_u_focus") if init_write_W_u_focus is None\
                                                                                       else init_write_W_u_focus
        self.read_head = MANNHeadGroup(control_state_dim=control_state_dim,
                                        memory_size=memory_size,
                                        memory_state_dim=memory_state_dim,
                                        k_smallest = k_smallest,
                                        num_heads = num_reads,
                                        gamma = self.gamma,
                                        is_write=False,
                                        name=self.name + "->read_head")
        self.write_head = MANNHeadGroup(control_state_dim=control_state_dim,
                                        memory_size=memory_size,
                                        memory_state_dim=memory_state_dim,
                                        k_smallest=k_smallest,
                                        num_heads=num_writes,
                                        gamma=self.gamma,
                                        is_write=True,
                                        init_W_r_focus=self.init_write_W_r_focus,
                                        init_W_u_focus=self.init_write_W_u_focus,
                                        name=self.name + "->write_head")
        self.read_counter = 0
        self.write_counter = 0
        self.memory = self.init_memory


    def read(self, control_input):
        """Read from the memory
        Parameters
            control_input: Shape (batch_size, control_state_dim)
            memory: Shape (batch_size, memory_size, memory_state_dim)
        ------------------------------------------------------
        Returns
            The read content --> Shape (batch_size,  memory_state_dim)
            The read focus   --> Shape (batch_size, num_heads, memory_size)
        """
        assert isinstance(control_input, mx.symbol.Symbol)
        read_content, read_focus, norm_key, norm_memory, similarity_score = self.read_head.read(control_input=control_input, memory=self.memory)
        self.read_counter += 1
        return read_content, read_focus, norm_key, norm_memory, similarity_score

    def write(self, control_input):
        """Write into the memory
        Parameters
            control_input: Shape (batch_size, control_state_dim)
            memory: Shape (batch_size, memory_size, memory_state_dim)
        ------------------------------------------------------
        Returns
            new_memory      --> Shape (batch_size, memory_size, memory_state_dim)
            The write focus --> Shape (batch_size, num_heads, memory_size)
        """
        assert isinstance(control_input, mx.symbol.Symbol)
        self.memory, erase_signal, add_signal, write_focus = self.write_head.write(control_input=control_input, memory=self.memory)
        ###self.memory, write_focus = self.write_head.write(control_input=control_input, memory=self.memory)
        return write_focus, erase_signal, add_signal, self.memory

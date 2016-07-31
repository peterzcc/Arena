from arena.operators import *
from arena.utils import *
import mxnet as mx

class MANNHead(object):
    def __init__(self, control_state_dim, memory_size, memory_state_dim, k_smallest,
                 init_W_r_focus=None, init_W_w_focus=None, init_W_u_focus=None,
                 is_write=False, name="MANNHead"):
        """
        :param control_state_dim:
        :param k_smallest:
        :param memory_size:
        :param memory_state_dim:
        :param init_read_W_r_focus: Shape (batch_size, num_reads, memory_size)
        :param init_read_W_w_focus: Shape (batch_size, num_reads, memory_size)
        :param init_read_W_u_focus: Shape (batch_size, num_reads, memory_size)
        :param init_write_W_r_focus: Shape (batch_size, num_write, memory_size)
        :param init_write_W_w_focus: Shape (batch_size, num_write, memory_size)
        :param init_write_W_u_focus: Shape (batch_size, num_write, memory_size)
        :param name:
        """
        self.name = name
        self.control_state_dim = control_state_dim
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim

        self.k_smallest = k_smallest

        self.key_weight = mx.sym.Variable(name=name + ":key_weight")
        self.key_bias = mx.sym.Variable(name=name + ":key_bias")
        self.alpha_weight = mx.sym.Variable(name=name + ":alpha_weight")
        self.alpha_bias = mx.sym.Variable(name=name + ":alpha_bias")
        self.beta_weight = mx.sym.Variable(name=name + ":beta_weight")
        self.beta_bias = mx.sym.Variable(name=name + ":beta_bias")
        self.gamma_weight = mx.sym.Variable(name=name + ":gamma_weight")
        self.gamma_bias = mx.sym.Variable(name=name + ":gamma_bias")

        self.is_write = is_write
        #if self.is_write:
        self.last_W_r_focus = init_W_r_focus if init_W_r_focus is not None \
            else mx.sym.Variable(name=name + ":init_W_r_focus")
        self.last_W_w_focus = init_W_w_focus if init_W_w_focus is not None \
            else mx.sym.Variable(name=name + ":init_W_w_focus")
        self.last_W_u_focus = init_W_u_focus if init_W_u_focus is not None \
            else mx.sym.Variable(name=name + ":init_W_u_focus")

        self.address_counter = 0

    # Controll is FNN
    def controller(self,control_input):
        """
            :param control_input: Shape (batch_size, control_state_dim)
            :return key: Shape (batch_size, memory_state_dim)
        """
        key = mx.sym.FullyConnected(data=control_input,
                                    num_hidden=self.memory_state_dim,
                                    weight=self.key_weight,
                                    bias=self.key_bias)  # Shape: (batch_size, memory_state_dim)
        key = mx.sym.Activation(data=key, act_type='tanh', name=self.name+":key")
        return key

    def addressing(self, control_input, memory):
        """
            :param  control_input: Shape (batch_size, control_state_dim)
            :param  memory:        Shape (batch_size, memory_size, memory_state_dim)
            :return W_r:           Shape (batch_size, memory_size)
            :return W_w:           Shape (batch_size, memory_size)
        """
        key = self.controller(control_input)
        # beta
        beta = mx.sym.FullyConnected(data=control_input,
                                     num_hidden=1,
                                     weight=self.beta_weight,
                                     bias=self.beta_bias)
        beta = mx.sym.Activation(data=beta, act_type='softrelu', name=self.name + ":beta")
        # Gamma
        gamma = mx.sym.FullyConnected(data=control_input,
                                      num_hidden=1,
                                      weight=self.gamma_weight,
                                      bias=self.gamma_bias)
        gamma = 1.0 + mx.sym.Activation(data=gamma, act_type='softrelu', name=self.name + ":gamma")
        # Alpha
        alpha = mx.sym.FullyConnected(data=control_input,
                                      num_hidden=1,
                                      weight=self.alpha_weight,
                                      bias=self.alpha_bias)
        alpha = mx.sym.Activation(data=alpha, act_type='sigmoid', name=self.name + ":alpha")
        # compute cosine similarity
        key = ArenaSym.normalize_channel(key, axis=1)
        memory = ArenaSym.normalize_channel(memory, axis=2)
        similarity_score = mx.sym.sum(mx.sym.broadcast_mul(mx.sym.expand_dims(key, axis=1), memory),
                                      axis=2)  # TODO Use batch_dot in the future
        # compute read weight
        W_r = mx.sym.SoftmaxActivation(mx.sym.broadcast_mul(beta, similarity_score))  # Shape: (batch_size, memory_size)
        # compute write weight
        last_W_lu = mx.sym.k_smallest_flags(self.last_W_u_focus, k=self.k_smallest)  # Shape (batch_size, memory_size)
        W_w = mx.sym.broadcast_mul(alpha, self.last_W_r_focus) + \
              mx.sym.broadcast_mul((1.0 - alpha), last_W_lu)
        W_u = mx.sym.broadcast_mul(gamma, self.last_W_u_focus) + W_r + W_w  # Shape (batch_size, memory_size)
        self.last_W_r_focus = W_r
        self.last_W_w_focus = W_w
        self.last_W_u_focus = W_u
        return W_r, W_w

    def read(self, control_input, memory):
        """
            :param control_input: Shape (batch_size, control_state_dim)
            :param memory: Shape (batch_size, memory_size, memory_state_dim)
            :return: Shape (batch_size, memory_state_dim)
        """
        read_focus, _ = self.addressing(control_input=control_input, memory=memory)
        content = mx.sym.sum(mx.sym.broadcast_mul(memory, mx.sym.expand_dims(read_focus, axis=2)),
                             axis=1)  # TODO Use batch_dot in the future
        return content, read_focus

    def write(self, control_input, memory):
        assert self.is_write
        key = self.controller(control_input)
        _, write_focus = self.addressing(control_input=control_input, memory=memory)
        new_memory = memory + \
                     mx.sym.broadcast_mul( mx.sym.expand_dims(write_focus, axis=2), mx.sym.expand_dims(key, axis=1) )
        return new_memory, write_focus



class MANN(object):
    def __init__(self, control_state_dim, memory_size, memory_state_dim, k_smallest,
                 num_reads, num_writes, init_memory=None,
                 init_read_W_r_focus=None, init_read_W_w_focus=None, init_read_W_u_focus=None,
                 init_write_W_r_focus=None, init_write_W_w_focus=None, init_write_W_u_focus=None,
                 name="MANN"):
        """
        :param control_state_dim:
        :param k_smallest:
        :param memory_size:
        :param memory_state_dim:
        :param num_reads:
        :param num_writes:
        :param init_memory: Shape (batch_size, memory_size, memory_state_dim)
        :param init_read_W_r_focus: Shape (batch_size, num_reads, memory_size)
        :param init_read_W_w_focus: Shape (batch_size, num_reads, memory_size)
        :param init_read_W_u_focus: Shape (batch_size, num_reads, memory_size)
        :param init_write_W_r_focus: Shape (batch_size, num_write, memory_size)
        :param init_write_W_w_focus: Shape (batch_size, num_write, memory_size)
        :param init_write_W_u_focus: Shape (batch_size, num_write, memory_size)
        :param name:
        """
        self.name = name
        self.control_state_dim = control_state_dim
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.k_smallest = k_smallest

        self.init_memory = mx.sym.Variable(self.name + ":init_memory") if init_memory is None\
                                                                       else init_memory
        self.init_read_W_r_focus = get_sym_list(init_read_W_r_focus,
                                                default_names=[(self.name + ":init_read_W_r_focus%d" %i)
                                                               for i in range(num_reads)])
        self.init_read_W_w_focus = get_sym_list(init_read_W_w_focus,
                                                default_names=[(self.name + ":init_read_W_w_focus%d" % i)
                                                               for i in range(num_reads)])
        self.init_read_W_u_focus = get_sym_list(init_read_W_u_focus,
                                                default_names=[(self.name + ":init_read_W_u_focus%d" % i)
                                                               for i in range(num_reads)])
        self.init_write_W_r_focus = get_sym_list(init_write_W_r_focus,
                                                default_names=[(self.name + ":init_write_W_r_focus%d" % i)
                                                               for i in range(num_writes)])
        self.init_write_W_w_focus = get_sym_list(init_write_W_w_focus,
                                                default_names=[(self.name + ":init_write_W_w_focus%d" % i)
                                                               for i in range(num_writes)])
        self.init_write_W_u_focus = get_sym_list(init_write_W_u_focus,
                                                default_names=[(self.name + ":init_write_W_u_focus%d" % i)
                                                               for i in range(num_writes)])
        self.read_heads = []
        self.write_heads = []
        self.read_heads = [MANNHead(control_state_dim=control_state_dim,
                                    memory_size=memory_size,
                                    memory_state_dim=memory_state_dim,
                                    k_smallest = k_smallest,
                                    is_write=False,
                                    init_W_r_focus=self.init_read_W_r_focus[i],
                                    init_W_w_focus=self.init_read_W_w_focus[i],
                                    init_W_u_focus=self.init_read_W_u_focus[i],
                                    name=self.name + "->read_head%d" %i)
                           for i in range(num_reads)]
        self.write_heads = [MANNHead(control_state_dim=control_state_dim,
                                     memory_size=memory_size,
                                     memory_state_dim=memory_state_dim,
                                     k_smallest=k_smallest,
                                     is_write=True,
                                     init_W_r_focus=self.init_write_W_r_focus[i],
                                     init_W_w_focus=self.init_write_W_w_focus[i],
                                     init_W_u_focus=self.init_write_W_u_focus[i],
                                     name=self.name + "->write_head%d" % i)
                            for i in range(num_writes)]
        #self.read_counter = 0
        #self.write_counter = 0
        self.memory = self.init_memory


    def read(self, control_input):
        """
        :param control_input  : Shape (batch_size, control_state_dim)
        :return read_content_l: Shape (batch_size, memory_state_dim)
        :return read_focus_l  : Shape (batch_size, memory_size)
        """
        assert isinstance(control_input, mx.symbol.Symbol)
        read_content_l = []
        read_focus_l = []
        for read_head in self.read_heads:
            content, read_focus = read_head.read(control_input=control_input, memory=self.memory)
            read_content_l.append(content)
            read_focus_l.append(read_focus)
        #self.read_counter += 1
        return read_content_l, read_focus_l

    def write(self, control_input):
        """
        :param control_input  : Shape (batch_size, control_state_dim)
        :return write_focus_l : Shape (batch_size, memory_size)
        """
        assert isinstance(control_input, mx.symbol.Symbol)
        write_focus_l = []
        for write_head in self.write_heads:
            self.memory, write_focus = write_head.write(control_input=control_input, memory=self.memory)
            write_focus_l.append(write_focus)
        #self.write_counter += 1
        return write_focus_l


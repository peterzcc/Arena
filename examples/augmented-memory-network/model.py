from arena.operators import *
from arena.utils import *
import mxnet as mx

class NTMHead(object):
    def __init__(self, memory_size, memory_state_dim, control_state_dim,
                 init_focus=None, is_write=False, num_shift=3, name="NTMHead", k_smallest=10):
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.control_state_dim = control_state_dim
        self.num_shift = num_shift
        self.name = name
        self.is_write = is_write

        self.key_weight = mx.sym.Variable(name=name + ":key_weight")
        self.key_bias = mx.sym.Variable(name=name + ":key_bias")
        self.alpha_weight = mx.sym.Variable(name=name + ":alpha_weight")
        self.alpha_bias = mx.sym.Variable(name=name + ":alpha_bias")
        self.beta_weight = mx.sym.Variable(name=name + ":beta_weight")
        self.beta_bias = mx.sym.Variable(name=name + ":beta_bias")
        #self.shift_weight = mx.sym.Variable(name=name + ":shift_weight")
        #self.shift_bias = mx.sym.Variable(name=name + ":shift_bias")
        self.gamma_weight = mx.sym.Variable(name=name + ":gamma_weight")
        self.gamma_bias = mx.sym.Variable(name=name + ":gamma_bias")
        self.init_focus = init_focus if init_focus is not None \
            else mx.sym.Variable(name=name + ":init_focus")
        if self.is_write:
            self.erase_signal_weight = mx.sym.Variable(name=name + ":erase_signal_weight")
            self.erase_signal_bias = mx.sym.Variable(name=name + ":erase_signal_bias")
            self.add_signal_weight = mx.sym.Variable(name=name + ":add_signal_weight")
            self.add_signal_bias = mx.sym.Variable(name=name + ":add_signal_bias")

        self.last_step_focus = self.init_focus
        self.last_weight_focus = self.init_focus # TODO need to change
        self.address_counter = 0
        self.k_smallest = k_smallest

    # Controll is FNN
    def controller(self,control_input):
        """
            :param control_input: Shape (batch_size, control_state_dim)
            :return: Shape (batch_size, memory_state_dim)
        """
        key = mx.sym.FullyConnected(data=control_input,
                                    num_hidden=self.memory_state_dim,
                                    weight=self.key_weight,
                                    bias=self.key_bias)  # Shape: (batch_size, memory_state_dim)
        key = mx.sym.Activation(data=key, act_type='tanh', name=self.name + ":key")
        return key
    #def compute_beta(self,control_input):

    def read_weight(self, control_input, memory):
        """
            :param control_input: Shape (batch_size, control_state_dim)
            :param memory: Shape (batch_size, memory_size, memory_state_dim)
            :return: Shape (batch_size, memory_size)
        """
        key = self.controller(control_input)
        # compute beta
        beta = mx.sym.FullyConnected(data=control_input,
                                     num_hidden=1,
                                     weight=self.beta_weight,
                                     bias=self.beta_bias)
        beta = mx.sym.Activation(data=beta, act_type='softrelu', name=self.name + ":beta")
        # compute cosine similarity
        key = ArenaSym.normalize_channel(key, axis=1)
        memory = ArenaSym.normalize_channel(memory, axis=2)
        similarity_score = mx.sym.sum(mx.sym.broadcast_mul(mx.sym.expand_dims(key, axis=1), memory),
                                      axis=2)  # TODO Use batch_dot in the future
        W_r = mx.sym.SoftmaxActivation(mx.sym.broadcast_mul(beta, similarity_score))  # Shape: (batch_size, memory_size)
        return W_r

    def read(self, control_input, memory):
        """
            :param control_input: Shape (batch_size, control_state_dim)
            :param memory: Shape (batch_size, memory_size, memory_state_dim)
            :return: Shape (batch_size, memory_state_dim)
        """
        W_r = self.read_weight(control_input=control_input, memory=memory)
        content = mx.sym.sum(mx.sym.broadcast_mul(memory, mx.sym.expand_dims(W_r, axis=2)),
                             axis=1)  # TODO Use batch_dot in the future
        return content

    def write_weight(self, control_input, memory):
        """
            :param control_input: Shape (batch_size, control_state_dim)
            :param memory: Shape (batch_size, memory_size, memory_state_dim)
            :return: Shape (batch_size, memory_size)
        """
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

        W_r = self.read_weight(control_input=control_input, memory=memory)
        W_u = mx.sym.Variable('W_u')
        W_u_new = mx.sym.broadcast_mul(gamma, W_u) + W_r + self.last_weight_focus # Shape (batch_size, memory_size) TODO batch_dot is not needed here
        W_lu = mx.sym.k_smallest_flags(W_u_new, k=self.k_smallest) # Shape (batch_size, memory_size)
        W_w = mx.sym.broadcast_mul(alpha, W_r) + mx.sym.broadcast_mul((1 - alpha), W_lu)

        self.last_weight_focus = W_w
        return W_w

    def write(self, control_input, memory):
        key = self.controller(control_input)
        W_w = self.write_weight(control_input=control_input, memory=memory)
        memory = memory + \
                 mx.sym.broadcast_mul( mx.sym.expand_dims(W_w, axis=2), mx.sym.expand_dims(key, axis=1) )



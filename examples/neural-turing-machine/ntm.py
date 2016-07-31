from arena.operators import *
from arena.utils import *
import mxnet as mx


class NTMHead(object):
    def __init__(self, memory_size, memory_state_dim, control_state_dim,
                 init_focus=None, is_write=False, num_shift=3, name="NTMHead"):
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.control_state_dim = control_state_dim
        self.num_shift = num_shift
        self.name = name
        self.is_write = is_write
        self.key_weight = mx.sym.Variable(name=name+ ":key_weight")
        self.key_bias = mx.sym.Variable(name=name + ":key_bias")
        self.gate_weight = mx.sym.Variable(name=name+ ":gate_weight")
        self.gate_bias = mx.sym.Variable(name=name + ":gate_bias")
        self.beta_weight = mx.sym.Variable(name=name+ ":beta_weight")
        self.beta_bias = mx.sym.Variable(name=name + ":beta_bias")
        self.shift_weight = mx.sym.Variable(name=name + ":shift_weight")
        self.shift_bias = mx.sym.Variable(name=name + ":shift_bias")
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
        self.address_counter = 0

    @property
    def params(self):
        if not self.is_write:
            return [self.key_weight, self.key_bias,
                    self.gate_weight, self.gate_bias,
                    self.beta_weight, self.beta_bias,
                    self.shift_weight, self.shift_bias,
                    self.gamma_weight, self.gamma_bias]
        else:
            return [self.key_weight, self.key_bias,
                    self.gate_weight, self.gate_bias,
                    self.beta_weight, self.beta_bias,
                    self.shift_weight, self.shift_bias,
                    self.gamma_weight, self.gamma_bias,
                    self.erase_signal_weight, self.erase_signal_bias,
                    self.add_signal_weight, self.add_signal_bias]

    @property
    def input_nodes(self):
        return [self.init_focus]

    def init_params(self):
        return None

    def addressing(self, control_input, memory):
        """

        :param control_input: Shape (batch_size, control_state_dim)
        :param memory: Shape (batch_size, memory_size, memory_state_dim)
        :return: Shape (batch_size, memory_size)
        """
        # Key
        key = mx.sym.FullyConnected(data=control_input,
                                 num_hidden=self.memory_state_dim,
                                 weight=self.key_weight,
                                 bias=self.key_bias) # Shape: (batch_size, memory_state_dim)
        key = mx.sym.Activation(data=key, act_type='tanh', name=self.name + ":key")
        # Beta
        beta = mx.sym.FullyConnected(data=control_input,
                                  num_hidden=1,
                                  weight=self.beta_weight,
                                  bias=self.beta_bias)
        beta = mx.sym.Activation(data=beta, act_type='softrelu', name=self.name + ":beta")
        # Gate
        gate = mx.sym.FullyConnected(data=control_input,
                                  num_hidden=1,
                                  weight=self.gate_weight,
                                  bias=self.gate_bias)
        gate = mx.sym.Activation(data=gate, act_type='sigmoid', name=self.name + ":gate")
        # Gamma
        gamma = mx.sym.FullyConnected(data=control_input,
                                   num_hidden=1,
                                   weight=self.gamma_weight,
                                   bias=self.gamma_bias)
        gamma = 1.0 + mx.sym.Activation(data=gamma, act_type='softrelu', name=self.name + ":gamma")
        # Shift
        shift = mx.sym.FullyConnected(data=control_input,
                                   num_hidden=self.num_shift,
                                   weight=self.shift_weight,
                                   bias=self.shift_bias)
        shift = mx.sym.SoftmaxActivation(shift, name=self.name + ":shift")
        # w_t^c = softmax(\beta K(k_t, M_t))
        key = ArenaSym.normalize_channel(key, axis=1)
        memory = ArenaSym.normalize_channel(memory, axis=2)
        similarity_score = mx.sym.sum(mx.sym.broadcast_mul(mx.sym.expand_dims(key, axis=1), memory), axis=2) #TODO Use batch_dot in the future
        wc = mx.sym.SoftmaxActivation(mx.sym.broadcast_mul(beta, similarity_score)) # Shape: (batch_size, memory_size)
        # w_t^g = g_t w_t^c + (1 - g_t) w_{t-1}
        wg = mx.sym.broadcast_mul(gate, wc) + mx.sym.broadcast_mul(1.0 - gate, self.last_step_focus)
        # w_t = w_t^g * s_t
        w = mx.sym.batch_cconv(wg, shift)
        # w_t = normalize(w_t ** r_t)
        w = ArenaSym.normalize_channel(mx.sym.broadcast_power(w, gamma), axis=1)
        self.last_step_focus = w
        self.address_counter += 1
        return w

    def read(self, control_input, memory):
        """
        :param control_input: Shape (batch_size, control_state_dim)
        :param memory: Shape (batch_size, memory_size, memory_state_dim)
        :return: the readed content --> Shape (batch_size, memory_state_dim)
        """
        assert not self.is_write
        read_focus = self.addressing(control_input=control_input, memory=memory)
        content = mx.sym.sum(mx.sym.broadcast_mul(memory, mx.sym.expand_dims(read_focus, axis=2)), axis=1)
        return content, read_focus

    def write(self, control_input, memory):
        """
        :param control_input: Shape (batch_size, control_state_dim)
        :param memory: Shape (batch_size, memory_size, memory_state_dim)
        :return:
        """
        assert self.is_write
        write_focus = self.addressing(control_input=control_input, memory=memory)
        erase_signal = mx.sym.FullyConnected(data=control_input,
                                       num_hidden=self.memory_state_dim,
                                       weight=self.erase_signal_weight,
                                       bias=self.erase_signal_bias)
        erase_signal = mx.sym.Activation(data=erase_signal, act_type='sigmoid',
                                   name=self.name + "_erase_signal")
        add_signal = mx.sym.FullyConnected(data=control_input,
                                     num_hidden=self.memory_state_dim,
                                     weight=self.add_signal_weight,
                                     bias=self.add_signal_bias)
        add_signal = mx.sym.Activation(data=add_signal, act_type='tanh',
                                 name=self.name + "_add_signal")
        new_memory = memory -\
                     memory * mx.sym.broadcast_mul(mx.sym.expand_dims(write_focus, axis=2),
                                                mx.sym.expand_dims(erase_signal, axis=1)) + \
                     mx.sym.broadcast_mul(mx.sym.expand_dims(write_focus, axis=2),
                                       mx.sym.expand_dims(add_signal, axis=1))
        return new_memory, erase_signal, add_signal, write_focus


class NTM(object):
    def __init__(self, num_reads, num_writes, memory_size, memory_state_dim, control_state_dim,
                 init_memory=None, init_read_focus=None, init_write_focus=None,
                 name="NTM"):
        """
        :param num_reads:
        :param num_writes:
        :param memory_size:
        :param memory_state_dim:
        :param control_state_dim:
        :param init_memory: Shape (batch_size, memory_size, memory_state_dim)
        :param init_read_focus: Shape (batch_size, num_reads, memory_size)
        :param init_write_focus: Shape (batch_size, num_write, memory_size)
        :param name:
        """
        self.name = name
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.control_state_dim = control_state_dim
        self.init_memory = mx.sym.Variable(self.name + ":init_memory") if init_memory is None\
                                                                    else init_memory

        self.init_read_focus = get_sym_list(init_read_focus,
                                             default_names=[(self.name + ":init_read_focus%d" %i)
                                                             for i in range(num_reads)])
        self.init_write_focus = get_sym_list(init_write_focus,
                                              default_names=[(self.name + ":init_write_focus%d" %i)
                                                             for i in range(num_writes)])
        self.read_heads = []
        self.write_heads = []

        self.read_heads = [NTMHead(control_state_dim=control_state_dim,
                                   memory_state_dim=memory_state_dim,
                                   memory_size=memory_size,
                                   is_write=False,
                                   init_focus=self.init_read_focus[i],
                                   name=self.name + "->read_head%d" %i)
                           for i in range(num_reads)]
        self.write_heads = [NTMHead(control_state_dim=control_state_dim,
                                    memory_state_dim=memory_state_dim,
                                    memory_size=memory_size,
                                    is_write=True,
                                    init_focus=self.init_write_focus[i],
                                    name=self.name + "->write_head%d" % i)
                            for i in range(num_writes)]
        self.read_counter = 0
        self.write_counter = 0
        self.memory = self.init_memory

    @property
    def params(self):
        ret = []
        for read_head in self.read_heads:
            ret.extend(read_head.params)
        for write_head in self.write_heads:
            ret.extend(write_head.params)
        return ret

    @property
    def input_nodes(self):
        ret = []
        for read_head in self.read_heads:
            ret.extend(read_head.input_nodes)
        for write_head in self.write_heads:
            ret.extend(write_head.input_nodes)
        ret.append(self.init_memory)
        return ret

    def read(self, control_input):
        assert isinstance(control_input, mx.symbol.Symbol)
        read_content_l = []
        read_focus_l = []
        for read_head in self.read_heads:
            content, read_focus = read_head.read(control_input=control_input, memory=self.memory)
            read_content_l.append(content)
            read_focus_l.append(read_focus)
        self.read_counter += 1
        return read_content_l, read_focus_l

    def write(self, control_input):
        assert isinstance(control_input, mx.symbol.Symbol)
        erase_signal_l = []
        add_signal_l = []
        write_focus_l = []
        for write_head in self.write_heads:
            self.memory, erase_signal, add_signal, write_focus = \
                write_head.write(control_input=control_input, memory=self.memory)
            erase_signal_l.append(erase_signal)
            add_signal_l.append(add_signal)
            write_focus_l.append(write_focus)
        return erase_signal_l, add_signal_l, write_focus_l

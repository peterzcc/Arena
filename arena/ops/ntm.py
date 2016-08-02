from arena.operators import *
import mxnet as mx


class NTMHeadGroup(object):
    def __init__(self, memory_size, memory_state_dim, control_state_dim,
                 num_heads=1, num_shift=3,
                 init_focus=None, is_write=False, name="NTMHead"):
        """

        Parameters
        ----------
        memory_size
        memory_state_dim
        control_state_dim
        num_heads
        num_shift
        init_focus
          Shape: (batch_size, num_heads, memory_size)
        is_write
        name
        """
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.control_state_dim = control_state_dim
        self.num_shift = num_shift
        self.num_heads = num_heads
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

        Parameters
        ----------
        control_input :
          Shape (batch_size, control_state_dim)
        memory :
          Shape (batch_size, memory_size, memory_state_dim)

        Returns
        -------
        Shape (batch_size, num_heads, memory_size)
        """
        # Key
        key = mx.sym.FullyConnected(data=control_input,
                                 num_hidden=self.num_heads * self.memory_state_dim,
                                 weight=self.key_weight,
                                 bias=self.key_bias) # Shape: (batch_size, num_heads * memory_state_dim)
        key = mx.sym.Reshape(key, shape=(-1, self.num_heads, self.memory_state_dim)) # Shape: (batch_size, num_heads, memory_state_dim)
        key = mx.sym.Activation(data=key, act_type='tanh', name=self.name + ":key")
        # Beta
        beta = mx.sym.FullyConnected(data=control_input,
                                  num_hidden=self.num_heads,
                                  weight=self.beta_weight,
                                  bias=self.beta_bias)
        beta = mx.sym.expand_dims(beta, axis=2) # Shape: (batch_size, num_heads, 1)
        beta = mx.sym.Activation(data=beta, act_type='softrelu', name=self.name + ":beta")
        # Gate
        gate = mx.sym.FullyConnected(data=control_input,
                                  num_hidden=self.num_heads,
                                  weight=self.gate_weight,
                                  bias=self.gate_bias)
        gate = mx.sym.expand_dims(gate, axis=2) # Shape: (batch_size, num_heads, 1)
        gate = mx.sym.Activation(data=gate, act_type='sigmoid', name=self.name + ":gate")
        # Gamma
        gamma = mx.sym.FullyConnected(data=control_input,
                                   num_hidden=self.num_heads,
                                   weight=self.gamma_weight,
                                   bias=self.gamma_bias)
        gamma = mx.sym.expand_dims(gamma, axis=2) # Shape: (batch_size, num_heads, 1)
        gamma = 1.0 + mx.sym.Activation(data=gamma, act_type='softrelu', name=self.name + ":gamma")
        # Shift
        shift = mx.sym.FullyConnected(data=control_input,
                                   num_hidden=self.num_heads * self.num_shift,
                                   weight=self.shift_weight,
                                   bias=self.shift_bias)
        shift = mx.sym.SoftmaxActivation(shift)
        shift = mx.sym.Reshape(shift, shape=(0, self.num_heads, self.num_shift),
                               name=self.name + ":shift")
        # w_t^c = softmax(\beta K(k_t, M_t))
        key = ArenaSym.normalize_channel(key, axis=2)
        memory = ArenaSym.normalize_channel(memory, axis=2)
        #similarity_score = mx.sym.sum(mx.sym.broadcast_mul(mx.sym.expand_dims(key, axis=1), memory), axis=2) #TODO Use batch_dot in the future
        similarity_score = mx.sym.batch_dot(key, mx.sym.SwapAxis(memory, dim1=1, dim2=2)) # Shape: (batch_size, num_heads, memory_size)
        wc = mx.sym.Reshape(mx.sym.SoftmaxActivation(mx.sym.Reshape(mx.sym.broadcast_mul(beta, similarity_score),
                                                            shape=(-1, self.memory_size))),
                            shape=(-1, self.num_heads, self.memory_size)) # Shape: (batch_size, num_heads, memory_size)
        # w_t^g = g_t w_t^c + (1 - g_t) w_{t-1}
        wg = mx.sym.broadcast_mul(gate, wc) + mx.sym.broadcast_mul(1.0 - gate, self.last_step_focus)
        # w_t = w_t^g * s_t
        w = mx.sym.batch_cconv(mx.sym.Reshape(wg, shape=(-1, self.memory_size)),
                               mx.sym.Reshape(shift, shape=(-1, self.num_shift)))
        w = mx.sym.Reshape(w, shape=(-1, self.num_heads, self.memory_size))
        # w_t = normalize(w_t ** r_t)
        w = ArenaSym.normalize_channel(mx.sym.broadcast_power(w, gamma), axis=2)
        self.last_step_focus = w
        self.address_counter += 1
        return w

    def read(self, control_input, memory):
        """Read from the memory

        Parameters
        ----------
        control_input:
            Shape (batch_size, control_state_dim)
        memory:
            Shape (batch_size, memory_size, memory_state_dim)
        Returns
        -------
        The read content --> Shape (batch_size, num_heads, memory_state_dim)
        """
        assert not self.is_write
        read_focus = self.addressing(control_input=control_input, memory=memory)
        content = mx.sym.batch_dot(read_focus, memory)
        return content, read_focus

    def write(self, control_input, memory):
        """Write to the memory

        Parameters
        ----------
        control_input:
            Shape (batch_size, control_state_dim)
        memory:
            Shape (batch_size, memory_size, memory_state_dim)
        Returns
        -------

        """
        assert self.is_write
        write_focus = self.addressing(control_input=control_input, memory=memory)
        erase_signal = mx.sym.FullyConnected(data=control_input,
                                       num_hidden=self.num_heads * self.memory_state_dim,
                                       weight=self.erase_signal_weight,
                                       bias=self.erase_signal_bias)
        erase_signal = mx.sym.Reshape(erase_signal,
                                      shape=(0, self.num_heads, self.memory_state_dim))
        erase_signal = mx.sym.Activation(data=erase_signal, act_type='sigmoid',
                                         name=self.name + "_erase_signal")
        add_signal = mx.sym.FullyConnected(data=control_input,
                                     num_hidden=self.num_heads * self.memory_state_dim,
                                     weight=self.add_signal_weight,
                                     bias=self.add_signal_bias)
        add_signal = mx.sym.Reshape(add_signal,
                                    shape=(0, self.num_heads, self.memory_state_dim))
        add_signal = mx.sym.Activation(data=add_signal, act_type='tanh',
                                       name=self.name + "_add_signal")
        erase_mult = 1 - mx.sym.batch_dot(mx.sym.Reshape(write_focus,
                                                         shape=(-1, self.memory_size, 1)),
                                          mx.sym.Reshape(erase_signal,
                                                         shape=(-1, 1, self.memory_state_dim)))
        if self.num_heads > 1:
            erase_mult = mx.sym.Reshape(erase_mult,
                                        shape=(-1, self.num_heads,
                                               self.memory_size, self.memory_state_dim))
            erase_mult = mx.sym.exp(mx.sym.sum(mx.sym.log(erase_mult), axis=1))
        aggre_add_signal = mx.sym.batch_dot(mx.sym.SwapAxis(write_focus, dim1=1, dim2=2),
                                            add_signal)
        new_memory = memory * erase_mult + aggre_add_signal
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

        self.init_read_focus = init_read_focus if init_read_focus is not None\
                                               else mx.sym.Variable(self.name + ":init_read_focus")
        self.init_write_focus = init_write_focus if init_write_focus is not None\
                                                 else mx.sym.Variable(self.name +
                                                                      ":init_write_focus")
        self.read_head = NTMHeadGroup(control_state_dim=control_state_dim,
                                      num_heads=num_reads,
                                      memory_state_dim=memory_state_dim,
                                      memory_size=memory_size,
                                      is_write=False,
                                      init_focus=self.init_read_focus,
                                      name=self.name + "->read_heads")
        self.write_head = NTMHeadGroup(control_state_dim=control_state_dim,
                                       num_heads=num_writes,
                                       memory_state_dim=memory_state_dim,
                                       memory_size=memory_size,
                                       is_write=True,
                                       init_focus=self.init_write_focus,
                                       name=self.name + "->write_heads")
        self.read_counter = 0
        self.write_counter = 0
        self.memory = self.init_memory

    @property
    def params(self):
        ret = []
        ret.extend(self.read_head.params)
        ret.extend(self.write_head.params)
        return ret

    @property
    def input_nodes(self):
        ret = []
        ret.extend(self.read_head.input_nodes)
        ret.extend(self.write_head.input_nodes)
        ret.append(self.init_memory)
        return ret

    def read(self, control_input):
        assert isinstance(control_input, mx.symbol.Symbol)
        content, read_focus = self.read_head.read(control_input=control_input, memory=self.memory)
        self.read_counter += 1
        return content, read_focus

    def write(self, control_input):
        assert isinstance(control_input, mx.symbol.Symbol)
        self.memory, erase_signal, add_signal, write_focus = \
            self.write_head.write(control_input=control_input, memory=self.memory)
        return erase_signal, add_signal, write_focus

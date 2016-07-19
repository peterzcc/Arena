from arena.operators import *
import mxnet as mx
from mxnet import sym


class NTMHead(object):
    def __init__(self, memory_size, memory_state_dim, control_state_dim,
                 init_focus=None, is_write=False, num_shift=3, name="NTMHead"):
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.control_state_dim = control_state_dim
        self.num_shift = num_shift
        self.name = name
        self.is_write = False
        self.key_weight = sym.Variable(name=name+ ":key_weight")
        self.key_bias = sym.Variable(name=name + ":key_bias")
        self.gate_weight = sym.Variable(name=name+ ":gate_weight")
        self.gate_bias = sym.Variable(name=name + ":gate_bias")
        self.beta_weight = sym.Variable(name=name+ ":beta_weight")
        self.beta_bias = sym.Variable(name=name + ":beta_bias")
        self.shift_weight = sym.Variable(name=name + ":shift_weight")
        self.shift_bias = sym.Variable(name=name + ":shift_bias")
        self.gamma_weight = sym.Variable(name=name + ":gamma_weight")
        self.gamma_bias = sym.Variable(name=name + ":gamma_bias")
        self.init_focus = init_focus if init_focus is not None \
                                                   else sym.Variable(name=name + ":init_focus")
        if self.is_write:
            self.erase_vec_weight = sym.Variable(name=name + ":erase_vec_weight")
            self.erase_vec_bias = sym.Variable(name=name + ":erase_vec_bias")
            self.add_vec_weight = sym.Variable(name=name + ":add_vec_weight")
            self.add_vec_bias = sym.Variable(name=name + ":add_vec_bias")

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
                    self.erase_vec_weight, self.erase_vec_bias,
                    self.add_vec_weight, self.add_vec_bias]

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
        key = sym.FullyConnected(data=control_input,
                                 num_hidden=self.memory_state_dim,
                                 weight=self.key_weight,
                                 bias=self.key_bias) # Shape: (batch_size, memory_state_dim)
        key = sym.Activation(data=key, act_type='tanh', name=self.name + "_key")
        beta = sym.FullyConnected(data=control_input,
                                  num_hidden=1,
                                  weight=self.beta_weight,
                                  bias=self.beta_bias)
        beta = sym.Activation(data=beta, act_type='softrelu', name=self.name + "_beta")
        gate = sym.FullyConnected(data=control_input,
                                  num_hidden=1,
                                  weight=self.gate_weight,
                                  bias=self.gate_bias)
        gate = sym.Activation(data=gate, act_type='sigmoid', name=self.name + "_gate")
        gamma = sym.FullyConnected(data=control_input,
                                   num_hidden=1,
                                   weight=self.gamma_weight,
                                   bias=self.gamma_bias)
        gamma = 1.0 + sym.Activation(data=gate, act_type='relu', name=self.name + "_gamma")
        shift = sym.FullyConnected(data=control_input,
                                   num_hidden=self.num_shift,
                                   weight=self.shift_weight,
                                   bias=self.shift_bias)
        shift = sym.SoftmaxActivation(shift, name=self.name + "_shift")
        # w_t^c = softmax(\beta K(k_t, M_t))
        key = ArenaSym.normalize_channel(key, axis=1)
        memory = ArenaSym.normalize_channel(memory, axis=2)
        similarity_score = sym.sum(sym.broadcast_mul(sym.expand_dims(key, axis=1), memory), axis=2) #TODO Use batch_dot in the future
        wc = sym.SoftmaxActivation(sym.broadcast_mul(beta, similarity_score)) # Shape: (batch_size, memory_size)
        # w_t^g = g_t w_t^c + (1 - g_t) w_{t-1}
        wg = sym.broadcast_mul(gate, wc) + sym.broadcast_mul(1 - gate, self.last_step_focus)
        # w_t = w_t^g * s_t
        w = sym.batch_cconv(wg, shift)
        # w_t = normalize(w_t ** r_t)
        w = ArenaSym.normalize_channel(sym.broadcast_pow(w, gamma), axis=1)
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
        read_weight = self.addressing(control_input=control_input, memory=memory)
        content = sym.sum(sym.broadcast_mul(memory, sym.expand_dims(read_weight, axis=2)), axis=1)
        return content, read_weight

    def write(self, control_input, memory):
        """
        :param control_input: Shape (batch_size, control_state_dim)
        :param memory: Shape (batch_size, memory_size, memory_state_dim)
        :return:
        """
        assert self.is_write
        write_weight = self.addressing(control_input=control_input, memory=memory)
        erase_vec = sym.FullyConnected(data=control_input,
                                       num_hidden=self.memory_state_dim,
                                       weight=self.erase_vec_weight,
                                       bias=self.erase_vec_bias)
        erase_vec = sym.Activation(data=erase_vec, act_type='sigmoid',
                                   name=self.name + "_erase_vec")
        add_vec = sym.FullyConnected(data=control_input,
                                     num_hidden=self.memory_state_dim,
                                     weight=self.add_vec_weight,
                                     bias=self.add_vec_bias)
        add_vec = sym.Activation(data=add_vec, act_type='tanh',
                                 name=self.name + "_add_vec")
        new_memory = memory -\
                     memory * sym.broadcast_mul(sym.expand_dims(write_weight, axis=2),
                                                sym.expand_dims(erase_vec, axis=1)) + \
                     sym.broadcast_mul(sym.expand_dims(write_weight, axis=2),
                                       sym.expand_dims(add_vec, axis=1))
        return new_memory, erase_vec, add_vec, write_weight


class NTM(object):
    def __init__(self, num_reads, num_writes, memory_size, memory_state_dim, control_state_dim,
                 init_memory=None, init_read_weights=None, init_write_weights=None,
                 name="NTM"):
        """
        :param num_reads:
        :param num_writes:
        :param memory_size:
        :param memory_state_dim:
        :param control_state_dim:
        :param init_memory: Shape (batch_size, memory_size, memory_state_dim)
        :param init_read_weights: Shape (batch_size, num_reads, memory_size)
        :param init_write_weights: Shape (batch_size, num_reads, memory_size)
        :param name:
        """
        self.name = name
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.control_state_dim = control_state_dim
        self.init_memory = sym.Variable(self.name + ":init_memory") if init_memory is None\
                                                                    else init_memory

        self.init_read_weights = sym.Variable(self.name + ":init_read_weights")\
                                 if init_read_weights is None else init_read_weights
        self.init_write_weights = sym.Variable(self.name + ":init_write_weights") \
                                  if init_write_weights is None else init_write_weights
        self.read_heads = []
        self.write_heads = []
        init_read_weights_split = mx.sym.SliceChannel(self.init_read_weights,
                                                      num_outputs=num_reads,
                                                      axis=1,
                                                      squeeze_axis=True,
                                                      name=self.name + "_split_init_read_weights")
        init_write_weights_split = mx.sym.SliceChannel(self.init_write_weights,
                                                       num_outputs=num_reads,
                                                       axis=1,
                                                       squeeze_axis=True,
                                                       name=self.name + "_split_init_write_weights")

        self.read_heads = [NTMHead(control_state_dim=control_state_dim,
                                   memory_state_dim=memory_state_dim,
                                   memory_size=memory_size,
                                   is_write=False,
                                   init_focus=init_read_weights_split[i],
                                   name=self.name + "->ReadHead%d" %i)
                           for i in range(num_reads)]
        self.write_heads = [NTMHead(control_state_dim=control_state_dim,
                                    memory_state_dim=memory_state_dim,
                                    memory_size=memory_size,
                                    is_write=True,
                                    init_focus=init_write_weights_split[i],
                                    name=self.name + "->WriteHead%d" % i)
                            for i in range(num_writes)]
        self.read_counter = 0
        self.write_counter = 0

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
        read_contents = []
        read_weights = []
        for read_head in self.read_heads:
            content, read_weight = read_head.read(control_input=control_input, memory=self.memory)
            read_contents.append(content)
            read_weights.append(read_weight)
        aggre_read_content = mx.sym.Concat(read_contents,
                                           name=self.name + "_aggre_read_content%d" %self.read_counter)
        aggre_read_weight = mx.sym.Concat(read_weights,
                                          name=self.name + "_aggre_read_weight%d" %self.read_counter)
        self.read_counter += 1
        return aggre_read_content, aggre_read_weight

    def write(self, control_input):
        erase_vecs = []
        add_vecs = []
        write_weights = []
        for write_head in self.write_heads:
            self.memory, erase_vec, add_vec, write_weight = \
                write_head.write(control_input=control_input, memory=self.memory)
            erase_vecs.append(erase_vec)
            add_vecs.append(add_vec)
            write_weights.append(write_weight)
        aggre_erase_vec = mx.sym.Concat(erase_vecs,
                                        name=self.name + "_aggre_erase_vec%d" %self.write_counter)
        aggre_add_vec = mx.sym.Concat(add_vecs,
                                      name=self.name + "_aggre_add_vec%d" %self.write_counter)
        aggre_write_weight = mx.sym.Concat(write_weights,
                                           name=self.name + "_aggre_write_weight%d" % self.write_counter)
        return aggre_erase_vec, aggre_add_vec, aggre_write_weight

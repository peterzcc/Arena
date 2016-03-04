import mxnet as mx
import numpy
from utils import *
import logging

logger = logging.getLogger(__name__)

#TODO Support RNN for sym, refer to the LSTM example
class Critic(object):
    """Critic, Differentiable Approximator for Q(s, a) or V(s)

    Parameters
    ----------
    data_shapes : dict
        The shapes of tensor variables
    sym: symbol of the critic network
    params:
    params_grad:
    aux_states:
    initializer:
    ctx:
    optimizer_params:
    name:

    """
    def __init__(self, data_shapes, sym, params=None, params_grad=None, aux_states=None,
                 initializer=mx.init.Uniform(0.07), ctx=mx.gpu(),
                 optimizer_params=None, name='CriticNet'):
        self.sym = sym
        self.ctx = ctx
        self.data_shapes = data_shapes.copy()
        self.name = name
        self.optimizer_params = optimizer_params.copy()
        if optimizer_params is not None:
            #TODO We may need to change here for distributed setting
            self.optimizer = mx.optimizer.create(**optimizer_params)
            self.updater = mx.optimizer.get_updater(self.optimizer)
        else:
            self.optimizer = None
            self.updater = None
        self.initializer=initializer
        if (params is None and params_grad is None and aux_states is None):
            assert initializer is not None, 'You must set the initializer if you donnot give the ' \
                                            'initial params!'
            arg_names = sym.list_arguments()
            aux_names = sym.list_auxiliary_states()
            param_names = list(set(arg_names) - set(self.data_shapes.keys()))
            arg_shapes, output_shapes, aux_shapes = sym.infer_shape(**self.data_shapes)
            print arg_shapes, self.data_shapes
            self.arg_name_shape = {k: s for k, s in zip(arg_names, arg_shapes)}
            self.params = {n: nd.empty(self.arg_name_shape[n], ctx=ctx) for n in param_names}
            self.params_grad = {n: nd.empty(self.arg_name_shape[n], ctx=ctx) for n in param_names}
            self.aux_states = {k: nd.empty(s, ctx=ctx) for k, s in zip(aux_names, aux_shapes)}
            for k, v in self.params.items():
                initializer(k, v)
        else:
            self.arg_name_shape = dict(data_shapes.items() + [(k, v.shape) for k, v in params.items()])
            self.params = params
            self.params_grad = params_grad
            self.aux_states = aux_states
        self.executor_pool = ExecutorBatchSizePool(ctx=self.ctx, sym=self.sym,
                                                   data_shapes=self.data_shapes,
                                                   params=self.params, params_grad=self.params_grad,
                                                   aux_states=self.aux_states)


    @property
    def default_batchsize(self):
        return self.data_shapes.values()[0].shape[0]

    """
    Compute the Q(s,a) or V(s) score
    """
    def calc_score(self, batch_size=default_batchsize, **input_dict):
        exe = self.executor_pool.get(batch_size)
        for k,v in input_dict.items():
            exe.arg_dict[k][:] = v
        exe.forward(is_train=False)
        for output in exe.outputs:
            output.wait_to_read()
        return exe.outputs

    def fit_target(self, batch_size=default_batchsize, **input_dict):
        assert self.updater is not None, "Updater not set! You may set critic_net.updater = ... " \
                                         "manually, or set the optimizer_params when you create" \
                                         "the object"
        exe = self.executor_pool.get(batch_size)
        for k, v in input_dict.items():
            exe.arg_dict[k][:] = v
        exe.forward(is_train=True)
        for output in exe.outputs:
            output.wait_to_read()
        exe.backward()
        for k in self.params:
            self.updater(index=k, grad=self.params_grad[k], weight=self.params[k])
        return exe.outputs

    """
    Can be used to calculate the gradient of Q(s,a) over a
    """
    def get_grads(self, keys, ctx=None, batch_size=default_batchsize, **input_dict):
        if len(input_dict) != 0 :
            exe = self.executor_pool.get(batch_size)
            for k, v in input_dict.items():
                exe.forward(is_train=True)
                exe.backward()
        all_grads = dict(self.params_grad.items() + self.executor_pool.inputs_grad_dict[batch_size].items())
        #TODO I'm not sure whether copy is needed here, need to test in the future
        if ctx is None:
            grads = {k : all_grads[k].copyto(all_grads[k].contenxt) for k in keys}
        else:
            grads = {k : all_grads[k].copyto(ctx) for k in keys}
        return grads

    def copyto(self, name=None, ctx=None):
        if ctx is None:
            ctx = self.ctx
        data_shapes = self.data_shapes.copy()
        sym = self.sym
        params = {k: v.copyto(ctx) for k, v in self.params.items()}
        params_grad = {k: v.copyto(ctx) for k, v in self.params_grad.items()}
        aux_states = None if self.aux_states is None else \
            {k: v.copyto(ctx) for k, v in self.aux_states.items()}
        optimizer_params = self.optimizer_params.copy()
        if name is None:
            name = self.name + '-copy-' + str(ctx)
        new_critic = Critic(data_shapes=data_shapes, sym=sym, params=params, params_grad=params_grad,
                        aux_states=aux_states, ctx=ctx, optimizer_params=optimizer_params,
                                                                            name=name)
        return new_critic

    def copy_params_to(self, dst):
        for k, v in self.params.items():
            dst.params[k][:] = v

    @property
    def total_param_num(self):
        return sum(v.size for v in self.params.values())

    def print_stat(self):
        logging.info("Name: %s" %self.name)
        assert self.params is not None, "Fatal Error!"
        logging.info("Params: ")
        for k, v in self.params.items():
            logging.info("   %s: %s" %(k, v.shape))
        if self.aux_states is None or 0 == len(self.aux_states):
            logging.info("Aux States: None")
        else:
            logging.info("Aux States: " + ' '.join(["%s:%s" %(str(k), str(v.shape)) for k, v in self.aux_states.items()]))
        logging.info("Total Parameter Num: " + str(self.total_param_num))
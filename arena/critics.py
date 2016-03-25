import mxnet as mx
import mxnet.ndarray as nd
import numpy
from utils import *
import os
import cPickle
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


# TODO Support RNN for sym, refer to the LSTM example
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

    def __init__(self, data_shapes, sym, params=None, aux_states=None,
                 initializer=mx.init.Uniform(0.07), ctx=mx.gpu(),
                 optimizer_params=None, name='CriticNet'):
        self.sym = sym
        self.ctx = ctx
        self.data_shapes = data_shapes.copy()
        self.name = name
        self.optimizer = None
        self.updater = None
        self.optimizer_params = None
        self.set_optimizer(optimizer_params)
        self.initializer = initializer
        if params is None:
            assert initializer is not None, 'We must set the initializer if we donnot give the ' \
                                            'initial params!'
            arg_names = sym.list_arguments()
            aux_names = sym.list_auxiliary_states()
            param_names = list(set(arg_names) - set(self.data_shapes.keys()))
            arg_shapes, output_shapes, aux_shapes = sym.infer_shape(**self.data_shapes)
            self.arg_name_shape = {k: s for k, s in zip(arg_names, arg_shapes)}
            self.params = {n: nd.empty(self.arg_name_shape[n], ctx=ctx) for n in param_names}
            self.params_grad = {n: nd.empty(self.arg_name_shape[n], ctx=ctx) for n in param_names}
            self.aux_states = {k: nd.empty(s, ctx=ctx) for k, s in zip(aux_names, aux_shapes)}
            for k, v in self.params.items():
                initializer(k, v)
        else:
            self.arg_name_shape = dict(
                data_shapes.items() + [(k, v.shape) for k, v in params.items()])
            self.params = {k: v.copyto(ctx) for k, v in params.items()}
            self.params_grad = {n: nd.empty(v.shape, ctx=ctx)
                                    for n, v in self.params.items()}
            if aux_states is not None:
                self.aux_states = {k: v.copyto(ctx) for k, v in aux_states.items()}
            else:
                self.aux_states = None
        self.executor_pool = ExecutorBatchSizePool(ctx=self.ctx, sym=self.sym,
                                                   data_shapes=self.data_shapes,
                                                   params=self.params, params_grad=self.params_grad,
                                                   aux_states=self.aux_states)

    def set_optimizer(self, optimizer_params=None):
        if optimizer_params is not None:
            # TODO We may need to change here for distributed setting
            self.optimizer_params = optimizer_params.copy()
            self.optimizer = mx.optimizer.create(**optimizer_params)
            self.updater = mx.optimizer.get_updater(self.optimizer)

    def save_params(self, dir_path="", epoch=None):
        param_saving_path = save_params(dir_path=dir_path, name=self.name, epoch=epoch,
                                            params=self.params,
                                            aux_states=self.aux_states)
        misc_saving_path = save_misc(dir_path=dir_path, epoch=epoch, name=self.name,
                                     data_shapes=self.data_shapes,
                                     optimizer_params=self.optimizer_params)
        logging.info('Saving %s, params: \"%s\", misc: \"%s\"',
                     self.name, param_saving_path, misc_saving_path)

    def load_params(self, name="", dir_path="", epoch=None):
        params, aux_states, param_loading_path = load_params(dir_path=dir_path, epoch=epoch, name=name)
        logging.info('Loading params from \"%s\" to %s' %(param_loading_path, self.name))
        for k, v in params.items():
            self.params[k][:] = v
        for k, v in aux_states.items():
            self.aux_states[k][:] = v

    @property
    def default_batchsize(self):
        return self.data_shapes.values()[0].shape[0]

    """
    Compute the Q(s,a) or V(s) score
    """
    def calc_score(self, batch_size=default_batchsize, **input_dict):
        exe = self.executor_pool.get(batch_size)
        #TODO `wait_to_read()` here seems unnecessary, remove it in the future!
        for v in self.params.values():
            v.wait_to_read()
        for k, v in input_dict.items():
            v.wait_to_read()
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
        for v in self.params.values():
            v.wait_to_read()
        for k, v in input_dict.items():
            v.wait_to_read()
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
        if len(input_dict) != 0:
            exe = self.executor_pool.get(batch_size)
            for k, v in input_dict.items():
                exe.forward(is_train=True)
                exe.backward()
        all_grads = dict(
            self.params_grad.items() + self.executor_pool.inputs_grad_dict[batch_size].items())
        # TODO I'm not sure whether copy is needed here, need to test in the future
        if ctx is None:
            grads = {k: all_grads[k].copyto(all_grads[k].contenxt) for k in keys}
        else:
            grads = {k: all_grads[k].copyto(ctx) for k in keys}
        return grads

    def copy(self, name=None, ctx=None):
        if ctx is None:
            ctx = self.ctx
        if name is None:
            name = self.name + '-copy-' + str(ctx)
        return Critic(data_shapes=self.data_shapes, sym=self.sym,
                      params=self.params,
                      aux_states=self.aux_states, ctx=ctx,
                      optimizer_params=self.optimizer_params, name=name)

    def copy_params_to(self, dst):
        for k, v in self.params.items():
            dst.params[k][:] = v
            #TODO `wait_to_read()` here seems unnecessary, remove it in the future!
            dst.params[k].wait_to_read()

    @property
    def total_param_num(self):
        return sum(v.size for v in self.params.values())

    def print_stat(self):
        logging.info("Name: %s" % self.name)
        assert self.params is not None, "Fatal Error!"
        logging.info("Params: ")
        for k, v in self.params.items():
            logging.info("   %s: %s" % (k, v.shape))
        if self.aux_states is None or 0 == len(self.aux_states):
            logging.info("Aux States: None")
        else:
            logging.info("Aux States: " + ' '.join(
                ["%s:%s" % (str(k), str(v.shape)) for k, v in self.aux_states.items()]))
        logging.info("Total Parameter Num: " + str(self.total_param_num))

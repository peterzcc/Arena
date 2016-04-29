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
class Base(object):
    """Basic wrapper for the symbols

    Parameters
    ----------
    data_shapes : dict
        The shapes of tensor variables
    sym: symbol of the network
    params:
    params_grad:
    aux_states:
    initializer:
    ctx:
    name:

    """

    def __init__(self, data_shapes, sym, params=None, aux_states=None,
                 initializer=mx.init.Uniform(0.07), ctx=mx.gpu(), name='Net'):
        self.sym = sym
        self.ctx = ctx
        self.data_shapes = data_shapes.copy()
        self.name = name
        self.initializer = initializer
        arg_names = sym.list_arguments()
        aux_names = sym.list_auxiliary_states()
        param_names = [n for n in arg_names if n not in self.data_shapes.keys()]
        arg_shapes, output_shapes, aux_shapes = sym.infer_shape(**self.data_shapes)
        self.arg_name_shape = OrderedDict([(k, s) for k, s in zip(arg_names, arg_shapes)])
        if params is None:
            self.params = OrderedDict([(n, nd.empty(self.arg_name_shape[n], ctx=ctx))
                                       for n in param_names])
            self.params_grad = OrderedDict([(n, nd.empty(self.arg_name_shape[n], ctx=ctx))
                                            for n in param_names])
            if len(self.params) > 0:
                assert initializer is not None, 'We must set the initializer if we donnot initialize' \
                                                'manually the free parameters of the network!!'
            for k, v in self.params.items():
                initializer(k, v)
        else:
            assert set(self.arg_name_shape.items()) == set(data_shapes.items() + [(k, v.shape)
                                                                     for k, v in params.items()])
            self.params = OrderedDict([(k, v.copyto(ctx)) for k, v in params.items()])
            self.params_grad = OrderedDict([(n, nd.empty(v.shape, ctx=ctx))
                                            for n, v in self.params.items()])
        if aux_states is not None:
            self.aux_states = OrderedDict([(k, v.copyto(ctx)) for k, v in aux_states.items()])
        else:
            self.aux_states = OrderedDict([(k, nd.empty(s, ctx=ctx))
                                           for k, s in zip(aux_names, aux_shapes)])
        self.acc_grad = None
        self.executor_pool = ExecutorDataShapePool(ctx=self.ctx, sym=self.sym,
                                                   data_shapes=self.data_shapes,
                                                   params=self.params, params_grad=self.params_grad,
                                                   aux_states=self.aux_states)

    def save_params(self, dir_path="", epoch=None):
        param_saving_path = save_params(dir_path=dir_path, name=self.name, epoch=epoch,
                                            params=self.params,
                                            aux_states=self.aux_states)
        misc_saving_path = save_misc(dir_path=dir_path, epoch=epoch, name=self.name,
                                     data_shapes=self.data_shapes)
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
    def internal_sym_names(self):
        return self.executor_pool.internal_syms.list_outputs()

    @property
    def default_batchsize(self):
        return self.data_shapes.values()[0].shape[0]

    def forward(self, batch_size=None, data_shapes=None, sym_name=None, is_train=False, **input_dict):
        exe = self.executor_pool.get(batch_size=batch_size, data_shapes=data_shapes,
                                     internal_sym_name=sym_name)
        if sym_name is not None:
            assert is_train is False, "We can only view the internal symbols using the " \
                                      "forward function!"
        #TODO `wait_to_read()` here seems unnecessary, remove it in the future!
        for v in self.params.values():
            v.wait_to_read()
        for k, v in input_dict.items():
            exe.arg_dict[k][:] = v
        exe.forward(is_train=is_train)
        for output in exe.outputs:
            output.wait_to_read()
        return exe.outputs

    def backward(self, batch_size=None, data_shapes=None, **arg_dict):
        exe = self.executor_pool.get(batch_size=batch_size,
                                     data_shapes=data_shapes)
        for k, v in arg_dict.items():
            exe.arg_dict[k][:] = v
        exe.backward()

    def update(self, updater, params_grad=None):
        if params_grad is None:
            params_grad = self.params_grad
        assert type(params_grad) is OrderedDict
        for ind, k in enumerate(self.params.keys()):
            updater(index=ind, grad=params_grad[k], weight=self.params[k])

    def update_acc_grad(self):
        if self.acc_grad is None:
            self.acc_grad = OrderedDict([(n, nd.zeros(v.shape, ctx=self.ctx))
                                         for n, v in self.params_grad.items()])
        for k, v in self.acc_grad.items():
            v[:] = v + self.params_grad[k]

    def reset_acc_grad(self):
        for v in self.acc_grad.values():
            v[:] = 0

    """
    Can be used to calculate the gradient of Q(s,a) over a
    """
    #TODO Test this part!
    def get_grads(self, keys, ctx=None, batch_size=None, data_shapes=None, **input_dict):
        if len(input_dict) != 0:
            exe = self.executor_pool.get(batch_size, data_shapes)
            for k, v in input_dict.items():
                exe.arg_dict[k][:] = v
                exe.forward(is_train=True)
                exe.backward()
        all_grads = OrderedDict(
            self.params_grad.items() + self.executor_pool.inputs_grad_dict[batch_size].items())
        # TODO I'm not sure whether copy is needed here, need to test in the future
        if ctx is None:
            grads = OrderedDict([(k, all_grads[k].copyto(all_grads[k].contenxt)) for k in keys])
        else:
            grads = OrderedDict([(k, all_grads[k].copyto(ctx)) for k in keys])
        return grads

    def copy(self, name=None, ctx=None):
        if ctx is None:
            ctx = self.ctx
        if name is None:
            name = self.name + '-copy-' + str(ctx)
        return Base(data_shapes=self.data_shapes, sym=self.sym,
                      params=self.params,
                      aux_states=self.aux_states, ctx=ctx, name=name)

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

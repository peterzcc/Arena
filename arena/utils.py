import mxnet as mx
import mxnet.ndarray as nd
import os
import numpy
import json

_ctx = mx.cpu()
_numpy_rng = numpy.random.RandomState(123456)


def get_default_ctx():
    return _ctx


def get_numpy_rng():
    return _numpy_rng


def get_saving_path(prefix="", epoch=None,):
    sym_saving_path = os.path.join('%s-symbol.json' % prefix)
    if epoch is not None:
        param_saving_path = os.path.join('%s-%05d.params' % (prefix, epoch))
    else:
        param_saving_path = os.path.join('%s.params' % prefix)
    misc_saving_path = os.path.join('%s-misc.json' % prefix)
    return sym_saving_path, param_saving_path, misc_saving_path


def save_params(dir_path="", epoch=None, name="", params=None, aux_states=None, ctx=mx.cpu()):
    prefix = os.path.join(dir_path, name)
    _, param_saving_path, _ = get_saving_path(prefix, epoch)
    if not os.path.isdir(dir_path) and not (dir_path == ""):
        os.makedirs(dir_path)
    save_dict = {('arg:%s' % k): v.copyto(ctx) for k, v in params.items()}
    save_dict.update({('aux:%s' % k): v.copyto(ctx) for k, v in aux_states.items()})
    nd.save(param_saving_path, save_dict)
    return param_saving_path


def save_misc(dir_path="", epoch=None, name="", **argdict):
    prefix = os.path.join(dir_path, name)
    _, _, misc_saving_path = get_saving_path(prefix, epoch)
    with open(misc_saving_path, 'w') as fp:
        json.dump(argdict, fp)
    return misc_saving_path


def load_params(dir_path="", epoch=None, name=""):
    prefix = os.path.join(dir_path, name)
    _, param_loading_path, _ = get_saving_path(prefix, epoch)
    save_dict = nd.load(param_loading_path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params, param_loading_path


def load_misc(dir_path="", epoch=None, name=""):
    prefix = os.path.join(dir_path, name)
    _, _, misc_saving_path = get_saving_path(prefix, epoch)
    with open(misc_saving_path, 'r') as fp:
        misc = json.load(fp)
    return misc

def update_to_kvstore(kv,params,params_grad):
    for paramIndex in range(len(params)):
        k=params.keys()[paramIndex]
        kv.push(paramIndex,params_grad[k],priority=-paramIndex)
        kv.pull(paramIndex,params[k],priority=-paramIndex)

class ExecutorBatchSizePool(object):
    def __init__(self, ctx, sym, data_shapes, params, params_grad, aux_states,share_execs=True):
        self.ctx = ctx
        self.sym = sym
        self.params = params
        self.params_grad = params_grad
        self.aux_states = aux_states
        self.data_dims = {}
        self.inputs_grad_dict = {}
        self.init_batch_size = data_shapes.values()[0][0]
        for k, v in data_shapes.items():
            self.data_dims[k] = v[1::]
            assert self.init_batch_size == v[0]
        self.exe_pool = {}
        self.share_execs = share_execs
        self.base_exe = self.get(self.init_batch_size)

    def get(self, batch_size=None):
        assert isinstance(batch_size, (int, long))
        if batch_size is None:
            batch_size = self.init_batch_size
        if batch_size in self.exe_pool:
            return self.exe_pool[batch_size]
        else:
            data_inputs = {k: mx.nd.empty((batch_size,) + s, ctx=self.ctx)
                           for k, s in self.data_dims.items()}
            inputs_grad = {k: mx.nd.empty((batch_size,) + s, ctx=self.ctx)
                           for k, s in self.data_dims.items()}
            self.inputs_grad_dict[batch_size] = inputs_grad
            if len(self.exe_pool) == 0:
                exe = self.sym.bind(ctx=self.ctx, args=dict(self.params, **data_inputs),
                                    args_grad=dict(self.params_grad.items() + inputs_grad.items()),
                                    aux_states=self.aux_states)
            else:
                if self.share_execs ==True:
                    exe = self.sym.bind(ctx=self.ctx, args=dict(self.params, **data_inputs),
                                        args_grad=dict(self.params_grad.items() + inputs_grad.items()),
                                        aux_states=self.aux_states, shared_exec=self.base_exe)
                else:
                    exe = self.sym.bind(ctx=self.ctx, args=dict(self.params, **data_inputs),
                                        args_grad=dict(self.params_grad.items() + inputs_grad.items()),
                                        aux_states=self.aux_states)
            self.exe_pool[batch_size] = exe
            return exe

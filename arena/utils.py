import mxnet as mx
import mxnet.ndarray as nd

class ExecutorBatchSizePool(object):
    def __init__(self, ctx, sym, data_shapes, params, params_grad, aux_states):
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
        self.hits = {}
        self.get(self.init_batch_size)

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
            exe = self.sym.bind(ctx=self.ctx, args=dict(self.params, **data_inputs),
                                args_grad=dict(self.params_grad.items() + inputs_grad.items()),
                                aux_states=self.aux_states)
            self.exe_pool[batch_size] = exe
            return exe
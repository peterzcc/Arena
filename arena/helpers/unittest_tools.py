# coding: utf-8
"""Tools for testing."""
from __future__ import absolute_import, print_function, division
import mxnet as mx
from mxnet.operator import NumpyOp
import numpy as np
from numpy.testing import assert_allclose, build_err_msg
import time
_rng = np.random.RandomState(1234)


def np_reduce(dat, axis, keepdims, numpy_reduce_func):
    """ Compatibility Reduce Function for Travis Numpy

    Parameters:
    -----------
    dat: np.ndarray
        Same as Numpy

    axis: None or int or list-like
        Same as Numpy

    keepdims: bool
        Same as Numpy

    numpy_reduce_func: function
        Numpy reducing function like `numpy.sum` or `numpy.max`
    """
    if isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis) if axis is not None else range(len(dat.shape))
    ret = dat
    for i in reversed(sorted(axis)):
        ret = numpy_reduce_func(ret, axis=i)
    if keepdims:
        keepdims_shape = list(dat.shape)
        for i in axis:
            keepdims_shape[i] = 1
        ret = ret.reshape(tuple(keepdims_shape))
    return ret


def reldiff(a, b):
    diff = np.sum(np.abs(a - b))
    norm = np.sum(np.abs(a)) + np.sum(np.abs(b))
    if diff == 0:
        return 0
    reldiff = diff / norm
    return reldiff


def numeric_grad(executor, locations, eps=1e-4):
    """ Class based on Theano's `theano.gradient.numeric_grad` [1]
    Calculates a numeric gradient via finite difference method.

    Parameters:
    -----------
    executor: `mxnet.executor.Executor`
        exectutor that computes the forward pass

    locations: list of numpy.ndarray or dict of str to numpy.ndarray
        Argument values used as locations to compute gradient

        - If type is list of numpy.ndarray, the position is in
          the same order of `executor.arg_arrays`.
        - If type is dict of str to numpy.ndarray, then it maps the name of arguments
          to the corresponding numpy.ndarray.
        - In either case, value of all the arguments must be provided.

    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """
    args = executor.arg_arrays
    assert isinstance(locations, (dict, list, tuple))
    if isinstance(locations, (list, tuple)):
        locations = {k: mx.nd.array(v, ctx=executor._ctx) for k, v in
                     zip(executor._symbol.list_arguments(), locations)}
    else:
        locations = {k: mx.nd.array(v, ctx=executor._ctx) for k, v in
                     locations.items()}
    for k, v in locations.items():
        executor.arg_dict[k][:] = v
    approx_grads = {k:np.zeros(v.shape) for k, v in locations.items()}

    executor.forward(is_train=False)
    f_x = executor.outputs[0].asnumpy()[0]
    for k, v in locations.items():
        old_value = v.copyto(v.context)
        for i in range(np.prod(v.shape)):
            # inplace update
            v.reshape((np.prod(v.shape), 1))[i] += eps
            # set initial states. Need to set all due to inplace operations
            for key, val in locations.items():
                executor.arg_dict[key][:] = val
            executor.forward(is_train=False)
            f_eps = executor.outputs[0].asnumpy()[0]
            approx_grads[k].ravel()[i] = (f_eps - f_x) / eps
            v.reshape((np.prod(v.shape), 1))[i] = old_value.reshape((np.prod(v.shape), 1))[i]

    return approx_grads


def check_numeric_gradient(sym, locations, ctx=mx.cpu(), grad_nodes=None, aux_states=None, rng=_rng,
                           numeric_eps=1e-4, check_eps=1e-2):
    """
    Verify an operation by checking backwards pass via
    finite difference method.

    Based on Theano's `theano.gradient.verify_grad` [1]

    Parameters:
    -----------
    sym : `mxnet.symbol.Symbol`
        Symbol containing op to test

    locations : list or tuple or dict
        Argument values used as locations to compute gradient

        - If type is list of numpy.ndarray, the position is in
          the same order of `sym.list_arguments()`.
        - If type is dict of str -> numpy.ndarray, then it maps the name of arguments
          to the corresponding numpy.ndarray.
        - In either case, value of all the arguments must be provided.

    ctx : Context, optional
        Check the gradient computation on the specified device

    grad_nodes : None or list or tuple, optional
        Names of the nodes to check gradient on

    aux_states : ist or tuple or dict, optional
        The auxiliary states required when generating the executor for the symbol

    rng : numpy.random.RandomState, optional

    numeric_eps : float, optional
        Delta for the finite difference method that approximates the gradient

    check_eps : float, optional
        relative error eps used when comparing numeric grad to symbolic grad

    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """

    def random_projection(shape):
        # random_projection should not have elements too small,
        # otherwise too much precision is lost in numerical gradient
        plain = rng.rand(*shape) + 0.1
        return plain
    assert isinstance(locations, (dict, list, tuple))
    if isinstance(locations, dict):
        assert set(sym.list_arguments()) == set(locations.keys())
    else:
        locations = {k:v for k,v in zip(sym.list_arguments(), locations)}

    if aux_states is not None:
        assert isinstance(aux_states, (dict, list, tuple))
        if isinstance(aux_states, (list, tuple)):
            aux_names = sym.list_auxiliary_states()
            aux_states = {k:v for k, v in zip(aux_names, aux_states)}

    if grad_nodes is None:
        grad_nodes = sym.list_arguments()
    else:
        assert isinstance(grad_nodes, (list, tuple))
        grad_nodes = list(grad_nodes)

    input_shape = {k:v.shape for k, v in locations.items()}
    arg_shape, out_shape, aux_shape = sym.infer_shape(**input_shape)
    proj = mx.sym.Variable("__random_proj")
    out = mx.sym.sum(sym * proj)
    out = mx.sym.MakeLoss(out)

    args = dict([(k, mx.nd.array(v, ctx=ctx)) for k, v in locations.items()] +
                    [("__random_proj", mx.nd.empty(out_shape[0], ctx=ctx))])
    args_grad = dict([(k, mx.nd.empty(locations[k].shape, ctx=ctx)) for k in grad_nodes] +
                    [("__random_proj", mx.nd.empty(out_shape[0], ctx=ctx))])

    executor = out.bind(ctx, args=args, args_grad=args_grad, aux_states=aux_states)

    locations = dict(locations.items() + [("__random_proj", random_projection(out_shape[0]))])
    inps = executor.arg_arrays
    if len(inps) != len(locations):
        raise ValueError("Executor arg_arrays and and locations len do not match."
                         "Got %d inputs and %d locationss"%(len(inps), len(locations)))
    for k, v in locations.items():
        executor.arg_dict[k][:] = v
    assert len(executor.outputs) == 1

    executor.forward(is_train=True)
    executor.backward()
    symbolic_grads = {k:executor.grad_dict[k].asnumpy() for k in grad_nodes}

    # refactor forward out of here as this no longer computes correct forward pass
    numeric_gradients = numeric_grad(executor, locations, eps=numeric_eps)
    for name in grad_nodes:
        fd_grad = numeric_gradients[name]
        sym_grad = symbolic_grads[name]
        rel = reldiff(fd_grad, sym_grad)
        if rel > check_eps:
            np.set_printoptions(threshold=4, suppress=True)
            msg = build_err_msg([fd_grad, sym_grad],
                                err_msg="In symbol \"%s\", "
                                        "numeric check failed for \"%s\". "
                                        "Rel Err=%f, Expected <=%f"
                                        %(sym.name, name, rel, check_eps),
                                names=["NUMERICAL", "BACKWARD"])
            raise Exception(msg)


def check_speed(sym, ctx=mx.cpu(), scale=1.0, N=100, grad_req=None, rng=_rng, **kwargs):
    if grad_req is None:
        grad_req = 'write'
    exe = sym.simple_bind(grad_req=grad_req, ctx=ctx, **kwargs)
    init = {k:np.random.normal(size=arr.shape, scale=scale) for k, arr in exe.arg_dict.items()}
    if "embedding_weight" in init:
        init['data'][:] = np.random.randint(0, init['embedding_weight'].shape[0], size=init['data'].shape)

    for name, iarr in init.items():
        exe.arg_dict[name][:] = iarr.astype(exe.arg_dict[name].dtype)

    # Warm up
    exe.forward(is_train=True)
    exe.backward(out_grads=exe.outputs)
    for output in exe.outputs:
        output.wait_to_read()
    # Test Forward + Backward
    tic = time.time()
    for i in range(N):
        exe.forward(is_train=True)
        exe.backward(out_grads=exe.outputs)
        for output in exe.outputs:
            output.wait_to_read()
    mx.nd.waitall()
    toc = time.time()
    forward_backward_time = (toc - tic) * 1.0 / N
    return forward_backward_time


def check_prediction_speed(sym, ctx=mx.cpu(), scale=1.0, N=100, rng=_rng, **kwargs):
    exe = sym.simple_bind(grad_req="null", ctx=ctx, **kwargs)
    init = {k: np.random.normal(size=arr.shape, scale=scale) for k, arr in exe.arg_dict.items()}
    if "embedding_weight" in init:
        init['data'][:] = np.random.randint(0, init['embedding_weight'].shape[0])

    for name, iarr in init.items():
        exe.arg_dict[name][:] = iarr.astype(exe.arg_dict[name].dtype)

    # Warm up
    exe.forward(is_train=False)
    for output in exe.outputs:
        output.wait_to_read()

    # Test Forward Only
    tic = time.time()
    for i in range(N):
        exe.forward(is_train=False)
        for output in exe.outputs:
            output.wait_to_read()
    toc = time.time()
    forward_time = (toc - tic) * 1.0 / N
    return forward_time


if __name__ == '__main__':
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    c = mx.sym.batch_cconv(a, b)
    a_npy = np.array([[1, 2, 3, 4, 5], [2,3,4,5,6]])
    b_npy = np.array([[1,2,3], [2,3,4]])

    exe = c.simple_bind(ctx=mx.gpu(), a=a_npy.shape, b=b_npy.shape)
    outputs = exe.forward(is_train=True, a=a_npy, b=b_npy)
    outputs[0].wait_to_read()
    print(outputs[0].asnumpy())

    check_numeric_gradient(sym=c, ctx=mx.gpu(),
                           locations=(a_npy, b_npy),
                           check_eps=0.01)

    sym_in = mx.sym.Variable('sym_in')
    sym_out = mx.sym.sum(sym_in, axis=(1, 3))
    # for i in range(2):
    #     check_numeric_gradient(sym=sym_out, ctx=mx.gpu(),
    #                            locations=(_rng.normal(0, 2, (10, 10, 10)),),
    #                            check_eps=0.05)
    print(check_speed(sym=sym_out, ctx=mx.gpu(), sym_in=(20, 20, 20, 20)))
    print(check_speed(sym=sym_out, ctx=mx.cpu(), sym_in=(20, 20, 20, 20)))
    print(check_prediction_speed(sym=sym_out, ctx=mx.gpu(), sym_in=(20, 20, 20, 20)))
    print(check_prediction_speed(sym=sym_out, ctx=mx.cpu(), sym_in=(20, 20, 20, 20)))
    tic = time.time()
    a = np.empty((20, 20))
    b = np.random.normal(0, 1, (20, 20, 20, 20))
    for i in range(100):
        a[:] = b.sum(axis=(1, 3))
    toc = time.time()
    print((toc - tic) * 1.0 / 100)

    print('Begin Benchmarking batch_cconv')
    print(check_speed(sym=c, ctx=mx.gpu(), a=(2048, 100), b=(2048, 3)))
    print(check_speed(sym=c, ctx=mx.cpu(), a=(2048, 100), b=(2048, 3)))
    print(check_speed(sym=c, ctx=mx.gpu(), a=(2048, 50), b=(2048, 5)))
    print(check_speed(sym=c, ctx=mx.cpu(), a=(2048, 50), b=(2048, 5)))
    print(check_speed(sym=c, ctx=mx.gpu(), a=(2048, 20), b=(2048, 3)))
    print(check_speed(sym=c, ctx=mx.cpu(), a=(2048, 20), b=(2048, 3)))


    a = mx.sym.Variable('a')
    b = mx.sym.Concat(a, a, num_args=2, dim=0)
    check_numeric_gradient(sym=b, ctx=mx.gpu(),
                           locations=(_rng.normal(0, 1, (10, 10, 10)),),
                           check_eps=0.01)

    # a = mx.sym.Variable('a')
    # b = mx.sym.transpose(a)
    # print('Begin Benchmarking transpose')
    # print(check_prediction_speed(sym=b, ctx=mx.gpu(), a=(100000, 128)))
    # print(check_prediction_speed(sym=b, ctx=mx.gpu(), a=(100000, 512)))
    # print(check_prediction_speed(sym=b, ctx=mx.gpu(), a=(500000, 1024)))


    data = mx.sym.Variable('data')
    embedding_weight = mx.sym.Variable('embedding_weight')
    embed = mx.sym.Embedding(data=data, weight=embedding_weight, input_dim=100000, output_dim=150)
    print('Begin Benchmarking embedding')
    print(check_speed(sym=embed, ctx=mx.gpu(), grad_req={'data': 'null', 'embedding_weight': 'add'},
                      data=(128, 100), embedding_weight=(100000, 150)))
    print(check_speed(sym=embed, ctx=mx.cpu(), grad_req={'data': 'null', 'embedding_weight': 'add'},
                      data=(128, 100), embedding_weight=(100000, 150)))
    print(check_prediction_speed(sym=embed, ctx=mx.gpu(), data=(128, 100), embedding_weight=(100000, 150)))
    print(check_prediction_speed(sym=embed, ctx=mx.cpu(), data=(128, 100), embedding_weight=(100000, 150)))



    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    c = mx.sym.batch_dot(a, b)
    d = mx.sym.broadcast_mul(a, b)
    d = mx.sym.sum(d, axis=2, keepdims=True)
    print('Begin Benchmarking batch_dot')
    print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 20, 100), b=(128, 100, 1)))
    print(check_speed(sym=c, ctx=mx.cpu(), a=(128, 20, 100), b=(128, 100, 1)))
    print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 100, 128), b=(128, 128, 1)))
    print(check_speed(sym=c, ctx=mx.cpu(), a=(128, 100, 128), b=(128, 128, 1)))
    print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 100, 500), b=(128, 500, 1)))
    print(check_speed(sym=c, ctx=mx.cpu(), a=(128, 100, 500), b=(128, 500, 1)))

    print('Begin Comparing batch_dot Versus broadcast + mul')
    print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 20, 100), b=(128, 100, 1)))
    print(check_speed(sym=d, ctx=mx.gpu(), a=(128, 20, 100), b=(128, 1, 100)))
    print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 100, 128), b=(128, 128, 1)))
    print(check_speed(sym=d, ctx=mx.gpu(), a=(128, 100, 128), b=(128, 1, 128)))
    print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 200, 500), b=(128, 500, 1)))
    print(check_speed(sym=d, ctx=mx.gpu(), a=(128, 200, 500), b=(128, 1, 500)))
    # print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 200, 1), b=(128, 1, 100)))
    # print(check_speed(sym=c, ctx=mx.cpu(), a=(128, 200, 1), b=(128, 1, 100)))
    # print(check_speed(sym=c, ctx=mx.gpu(), a=(128, 200, 100), b=(128, 100, 100)))
    # print(check_speed(sym=c, ctx=mx.cpu(), a=(128, 200, 100), b=(128, 100, 100)))
    # print(check_speed(sym=c, ctx=mx.gpu(), a=(16, 200, 100), b=(16, 100, 100)))
    # print(check_speed(sym=c, ctx=mx.cpu(), a=(16, 200, 100), b=(16, 100, 100)))
    #

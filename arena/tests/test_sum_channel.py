import mxnet as mx
import mxnet.ndarray as nd
import numpy


def test2():
    a = mx.symbol.Variable('a')
    c = mx.symbol.Variable('c')
    b = mx.symbol.sum_mid_internal(a)
    d = mx.symbol.sum(mx.symbol.square(b - c))
    d = mx.symbol.MakeLoss(mx.symbol.Reshape(d, shape=(1, 1)))
    a_npy = numpy.random.rand(120, 111, 12)
    c_npy = numpy.random.rand(120, 12)
    a_grad = nd.empty((120, 111, 12))

    a_ndarray = nd.array(a_npy)
    net = d.bind(mx.gpu(), args={'a': a_ndarray, 'c': nd.array(c_npy)},
                 args_grad={'a': a_grad})
    lr = 0.001
    for i in range(100):
        net.forward(is_train=True)
        loss = net.outputs[0].asnumpy()
        print loss
        net.backward()
        a_ndarray -= lr * a_grad


def test1():
    a = mx.symbol.Variable('a')
    b = mx.symbol.sum_mid_internal(a)

    a_npy = numpy.random.rand(120,111,12)

    a_grad = nd.empty((120, 111, 12))
    b_grad_npy = numpy.random.rand(120,12)

    net = b.bind(mx.gpu(), args={'a': nd.array(a_npy)},
             args_grad={'a': a_grad})
    net.forward(is_train=True)

    print numpy.square(net.outputs[0].asnumpy() - a_npy.sum(axis=1)).sum()

    net.backward(out_grads=nd.array(b_grad_npy))
    print numpy.square(a_grad.asnumpy() - b_grad_npy.reshape((120, 1, 12))).sum()
test1()
test2()
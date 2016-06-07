import numpy
import mxnet as mx
import mxnet.ndarray as nd
from arena import Base

def complex_mul(a, b):
    return a*b

def complex_mul_grad(out, a, b):
    return numpy.conj(b) * out, numpy.conj(a) * out

def complex_div(a, b):
    return a/b

def complex_div_grad(out_grad, a, b):
    out = a/b
    return b * out_grad / (b.real**2 + b.imag**2), \
           (a * numpy.conj(out_grad) - 2*(out.real * out_grad.real + out.imag * out_grad.imag)*b)/ \
           (b.real ** 2 + b.imag ** 2)

def loss(o):
    return 0.5 * (o.real **2 + o.imag **2)

def test_numpy():
    def gradient_testing_cc(out_f, grad_f, a, b):
        grad_a, grad_b = grad_f(out_f(a, b), a, b)
        eps = 1E-6
        print (loss(out_f(a + eps, b)) - loss(out_f(a, b))) / eps - grad_a.real
        print (loss(out_f(a + eps * 1j, b)) - loss(out_f(a, b))) / eps - grad_a.imag
        print (loss(out_f(a, b + eps)) - loss(out_f(a, b))) / eps - grad_b.real
        print (loss(out_f(a, b + eps * 1j)) - loss(out_f(a, b))) / eps - grad_b.imag
    def gradient_testing_cr(out_f, grad_f, a, b):
        b = b.real
        grad_a, grad_b = grad_f(out_f(a, b), a, b)
        eps = 1E-6
        print (loss(out_f(a + eps, b)) - loss(out_f(a, b))) / eps - grad_a.real
        print (loss(out_f(a + eps * 1j, b)) - loss(out_f(a, b))) / eps - grad_a.imag
        print (loss(out_f(a, b + eps)) - loss(out_f(a, b))) / eps - grad_b.real
    def gradient_testing_rc(out_f, grad_f, a, b):
        a = a.real
        grad_a, grad_b = grad_f(out_f(a, b), a, b)
        eps = 1E-6
        print (loss(out_f(a + eps, b)) - loss(out_f(a, b))) / eps - grad_a.real
        print (loss(out_f(a, b + eps)) - loss(out_f(a, b))) / eps - grad_b.real
        print (loss(out_f(a, b + eps * 1j)) - loss(out_f(a, b))) / eps - grad_b.imag

    def grad_testing_conj(a):
        eps = 1E-6
        print (loss(numpy.conj(a + eps)) - loss(numpy.conj(a))) / eps - a.real
        print (loss(numpy.conj(a + eps * 1j)) - loss(numpy.conj(a))) / eps - a.imag

    def grad_testing_abs_square(a):
        eps = 1E-6
        print (loss(abs(a + eps) ** 2) - loss(abs(a) ** 2)) / eps - 2 * (abs(a) ** 2 * a).real
        print (loss(abs(a + eps * 1j) ** 2) - loss(abs(a) ** 2)) / eps - 2 * (abs(a) ** 2 * a).imag

    a = numpy.random.rand() + numpy.random.rand() * 1j
    b = numpy.random.rand() + numpy.random.rand() * 1j
    print 'mul_cc:'
    gradient_testing_cc(complex_mul, complex_mul_grad, a, b)
    print 'mul_cr:'
    gradient_testing_cr(complex_mul, complex_mul_grad, a, b)
    print 'mul_rc:'
    gradient_testing_rc(complex_mul, complex_mul_grad, a, b)
    print 'div_cc:'
    gradient_testing_cc(complex_div, complex_div_grad, a, b)
    print 'div_cr:'
    gradient_testing_cr(complex_div, complex_div_grad, a, b)
    print 'div_rc:'
    gradient_testing_rc(complex_div, complex_div_grad, a, b)
    print 'conj:'
    grad_testing_conj(a)
    print 'complex_abs_square'
    grad_testing_abs_square(a)

def test_mxnet_binary(test_operation, typ):
    if 'div' == test_operation:
        numpy_outf = complex_div
        numpy_gradf = complex_div_grad
        if typ == 'rc':
            test_sym = mx.symbol.complex_div_rc
        elif typ == 'cc':
            test_sym = mx.symbol.complex_div_cc
        elif typ == 'cr':
            test_sym = mx.symbol.complex_div_cr
    else:
        numpy_outf = complex_mul
        numpy_gradf = complex_mul_grad
        if typ == 'rc':
            test_sym = mx.symbol.complex_mul_rc
        elif typ == 'cc':
            test_sym = mx.symbol.complex_mul_cc
        elif typ == 'cr':
            test_sym = mx.symbol.complex_mul_cr
    a = mx.symbol.Variable('a')
    b = mx.symbol.Variable('b')
    c = test_sym(a, b)
    base_complex_shape = (10, 10, 6)
    base_real_shape = (10, 10, 3)
    if 'cc' == typ:
        data_shapes = {'a': base_complex_shape, 'b': base_complex_shape}
        a_complex_npy = numpy.random.rand(*base_real_shape) + \
                        numpy.random.rand(*base_real_shape) * 1j
        b_complex_npy = numpy.random.rand(*base_real_shape) + \
                        numpy.random.rand(*base_real_shape) * 1j
        a_npy = numpy.empty(data_shapes['a'])
        b_npy = numpy.empty(data_shapes['b'])
        a_npy[:, :, ::2] = a_complex_npy.real
        a_npy[:, :, 1::2] = a_complex_npy.imag
        b_npy[:, :, ::2] = b_complex_npy.real
        b_npy[:, :, 1::2] = b_complex_npy.imag
        net = Base(data_shapes=data_shapes, sym=c)
        outputs = net.forward(a=a_npy, b=b_npy)
        out_grad = numpy.random.rand(*data_shapes['a'])
        print numpy.square(
            outputs[0].asnumpy()[:, :, ::2] - numpy_outf(a_complex_npy, b_complex_npy).real).sum()
        print numpy.square(
            outputs[0].asnumpy()[:, :, 1::2] - numpy_outf(a_complex_npy, b_complex_npy).imag).sum()
        net.backward(out_grads=[nd.array(out_grad, ctx=mx.gpu())])
        a_grad_npy, b_grad_npy = numpy_gradf(out_grad[:, :, ::2] + out_grad[:, :, 1::2] * 1j,
                                             a_complex_npy, b_complex_npy)
        print numpy.square(net.exe.grad_dict['a'].asnumpy()[:, :, ::2] - a_grad_npy.real).sum()
        print numpy.square(net.exe.grad_dict['a'].asnumpy()[:, :, 1::2] - a_grad_npy.imag).sum()
        print numpy.square(net.exe.grad_dict['b'].asnumpy()[:, :, ::2] - b_grad_npy.real).sum()
        print numpy.square(net.exe.grad_dict['b'].asnumpy()[:, :, 1::2] - b_grad_npy.imag).sum()
    elif 'rc' == typ:
        data_shapes = {'a':base_real_shape, 'b':base_complex_shape}
        a_complex_npy = numpy.random.rand(*base_real_shape)
        b_complex_npy = numpy.random.rand(*base_real_shape) + \
                        numpy.random.rand(*base_real_shape) * 1j
        a_npy = numpy.empty(data_shapes['a'])
        b_npy = numpy.empty(data_shapes['b'])
        a_npy = a_complex_npy
        b_npy[:, :, ::2] = b_complex_npy.real
        b_npy[:, :, 1::2] = b_complex_npy.imag
        net = Base(data_shapes=data_shapes, sym=c)
        outputs = net.forward(a=a_npy, b=b_npy)
        out_grad = numpy.random.rand(*data_shapes['b'])
        print numpy.square(
            outputs[0].asnumpy()[:, :, ::2] - numpy_outf(a_complex_npy, b_complex_npy).real).sum()
        print numpy.square(
            outputs[0].asnumpy()[:, :, 1::2] - numpy_outf(a_complex_npy, b_complex_npy).imag).sum()
        net.backward(out_grads=[nd.array(out_grad, ctx=mx.gpu())])
        a_grad_npy, b_grad_npy = numpy_gradf(out_grad[:, :, ::2] + out_grad[:, :, 1::2] * 1j,
                                             a_complex_npy, b_complex_npy)
        print numpy.square(net.exe.grad_dict['a'].asnumpy()- a_grad_npy.real).sum()
        print numpy.square(net.exe.grad_dict['b'].asnumpy()[:, :, ::2] - b_grad_npy.real).sum()
        print numpy.square(net.exe.grad_dict['b'].asnumpy()[:, :, 1::2] - b_grad_npy.imag).sum()
    else:
        data_shapes = {'a': base_complex_shape, 'b': base_real_shape}
        a_complex_npy = numpy.random.rand(*base_real_shape) + \
                        numpy.random.rand(*base_real_shape) * 1j
        b_complex_npy = numpy.random.rand(*data_shapes['b'])
        a_npy = numpy.empty(data_shapes['a'])
        b_npy = numpy.empty(data_shapes['b'])
        a_npy[:, :, ::2] = a_complex_npy.real
        a_npy[:, :, 1::2] = a_complex_npy.imag
        b_npy = b_complex_npy.real
        net = Base(data_shapes=data_shapes, sym=c)
        outputs = net.forward(a=a_npy, b=b_npy)
        out_grad = numpy.random.rand(*data_shapes['a'])
        print numpy.square(
            outputs[0].asnumpy()[:, :, ::2] - numpy_outf(a_complex_npy, b_complex_npy).real).sum()
        print numpy.square(
            outputs[0].asnumpy()[:, :, 1::2] - numpy_outf(a_complex_npy, b_complex_npy).imag).sum()
        net.backward(out_grads=[nd.array(out_grad, ctx=mx.gpu())])
        a_grad_npy, b_grad_npy = numpy_gradf(out_grad[:, :, ::2] + out_grad[:, :, 1::2] * 1j,
                                             a_complex_npy, b_complex_npy)
        print numpy.square(net.exe.grad_dict['a'].asnumpy()[:, :, ::2] - a_grad_npy.real).sum()
        print numpy.square(net.exe.grad_dict['a'].asnumpy()[:, :, 1::2] - a_grad_npy.imag).sum()
        print numpy.square(net.exe.grad_dict['b'].asnumpy()- b_grad_npy.real).sum()

def test_mxnet_conj():
    a = mx.symbol.Variable('a')
    b = mx.symbol.conj(a)
    base_shape = (2, 10)
    data_shapes = {'a': base_shape}
    a_npy = numpy.random.rand(*base_shape)
    out_grad_npy = numpy.random.rand(*base_shape)
    net = Base(sym=b, data_shapes=data_shapes)
    outputs = net.forward(is_train=True, a=a_npy)
    print 'conj:'
    print numpy.square(outputs[0].asnumpy()[:, ::2] - a_npy[:, ::2]).sum()
    print numpy.square(outputs[0].asnumpy()[:, 1::2] + a_npy[:, 1::2]).sum()
    net.backward(out_grads=[nd.array(out_grad_npy, ctx=mx.gpu())])
    print numpy.square(net.exe.grad_dict['a'].asnumpy()[:, ::2] - out_grad_npy[:, ::2]).sum()
    print numpy.square(net.exe.grad_dict['a'].asnumpy()[:, 1::2] + out_grad_npy[:, 1::2]).sum()

def test_mxnet_complex_abs_square():
    a = mx.symbol.Variable('a')
    b = mx.symbol.complex_abs_square(a)
    base_shape = (2, 10)
    base_real_shape = (2, 5)
    data_shapes = {'a': base_shape}
    a_npy = numpy.random.rand(*base_shape)
    out_grad_npy = numpy.random.rand(*base_real_shape)
    net = Base(sym=b, data_shapes=data_shapes)
    outputs = net.forward(is_train=True, a=a_npy)
    print 'complex_abs_square:'
    print numpy.square(outputs[0].asnumpy() - (a_npy[:,::2]**2 + a_npy[:, 1::2]**2)).sum()
    net.backward(out_grads=[nd.array(out_grad_npy, ctx=mx.gpu())])
    print numpy.square(net.exe.grad_dict['a'].asnumpy()[:, ::2] - 2 * out_grad_npy * a_npy[:, ::2]).sum()
    print numpy.square(
        net.exe.grad_dict['a'].asnumpy()[:, 1::2] - 2 * out_grad_npy * a_npy[:, 1::2]).sum()

test_numpy()
test_mxnet_binary('div', 'cc')
test_mxnet_binary('div', 'rc')
test_mxnet_binary('div', 'cr')
test_mxnet_binary('mul', 'cc')
test_mxnet_binary('mul', 'cr')
test_mxnet_binary('mul', 'rc')
test_mxnet_conj()
test_mxnet_complex_abs_square()
from arena.operators import *
from arena import Base
import matplotlib.pyplot as plt
import time

def logmog(prob, mean, var, sample, score):
    sample = sample.reshape(mean.shape[0], 1, mean.shape[2])
    ele_likelihood = prob * numpy.exp((- numpy.square(mean - sample) / var / 2 - 0.5 * numpy.log(var)).sum(axis=2))
    with numpy.errstate(divide='raise'):
        try:
            ret = - (numpy.nan_to_num(numpy.log(ele_likelihood.sum(axis=1)))*score).sum()
            return ret
        except :
            print ele_likelihood


def mog_sample_test():
    batch_size = 2
    num_centers = 8*8
    sample_dim = 2
    prob_npy = get_numpy_rng().rand(batch_size, num_centers)
    mean_npy = get_numpy_rng().rand(batch_size, num_centers, sample_dim)*10
    var_npy = get_numpy_rng().rand(batch_size, num_centers, sample_dim)*2
    prob_npy = prob_npy / prob_npy.sum(axis=1).reshape(prob_npy.shape[0], 1)
    score_npy = get_numpy_rng().rand(batch_size,)
    siz = numpy.sqrt(num_centers).astype(numpy.int32)
    for i in range(num_centers):
        mean_npy[0, i, 0] = (i % siz).astype(numpy.float32) / siz * 10
        mean_npy[0, i, 1] = (i / siz).astype(numpy.float32) / siz * 10
    total_sample_num = 500000
    print 'prob_npy:', prob_npy
    print 'mean_npy:', mean_npy
    print 'var_npy:', var_npy

    def speed_npy_test():
        sample_npy = numpy.empty((total_sample_num, mean_npy.shape[0], mean_npy.shape[2]), dtype=numpy.float32)
        for i in range(total_sample_num):
            sample_npy[i, :, :] = sample_mog(prob=prob_npy, mean=mean_npy, var=var_npy, rng=get_numpy_rng())
        plt.hist2d(sample_npy[:, 1, 0], sample_npy[:, 1, 1], (200, 200), cmap=plt.cm.jet)
        plt.colorbar()
        plt.show()

    def speed_mxnet_test():
        prob = mx.symbol.Variable('prob')
        mean = mx.symbol.Variable('mean')
        var = mx.symbol.Variable('var')
        score = mx.symbol.Variable('score')
        out = mx.symbol.Custom(prob=prob, mean=mean, var=var, score=score, name='policy', op_type='LogMoGPolicy')
        data_shapes = {'prob': (batch_size, num_centers), 'mean': (batch_size, num_centers, sample_dim),
                       'var': (batch_size, num_centers, sample_dim), 'score': (batch_size,)}

        net = Base(sym=out, data_shapes=data_shapes, ctx=mx.cpu())
        sample_npy = numpy.empty((total_sample_num, mean_npy.shape[0], mean_npy.shape[2]), dtype=numpy.float32)
        for i in range(total_sample_num):
            if 0 == i:
                sample_npy[i, :, :] = net.forward(is_train=False, prob=prob_npy, mean=mean_npy,
                                                  var=var_npy, score=score_npy)[0].asnumpy()
            else:
                sample_npy[i, :, :] = net.forward(is_train=False)[0].asnumpy()
        plt.hist2d(sample_npy[:, 1, 0], sample_npy[:, 1, 1], (200, 200), cmap=plt.cm.jet)
        plt.colorbar()
        plt.show()

    begin = time.time()
    speed_npy_test()
    end = time.time()
    print 'numpy:', end - begin

    begin = time.time()
    speed_mxnet_test()
    end = time.time()
    print 'mxnet:', end - begin


def mog_backward_test(batch_size=5, num_centers=11, sample_dim=33):
    prob = mx.symbol.Variable('prob')
    mean = mx.symbol.Variable('mean')
    var = mx.symbol.Variable('var')
    score = mx.symbol.Variable('score')
    out = mx.symbol.Custom(prob=prob, mean=mean, var=var, score=score, name='policy', op_type='LogMoGPolicy', implicit_backward=False)
    data_shapes = {'prob': (batch_size, num_centers), 'mean': (batch_size, num_centers, sample_dim),
                   'var': (batch_size, num_centers, sample_dim), 'score': (batch_size,),
                   'policy_backward_action': (batch_size, sample_dim)}
    net = Base(sym=out, data_shapes=data_shapes, ctx=mx.cpu())

    prob_npy = get_numpy_rng().rand(batch_size, num_centers)
    mean_npy = get_numpy_rng().rand(batch_size, num_centers, sample_dim) * 1 + 5
    var_npy = get_numpy_rng().rand(batch_size, num_centers, sample_dim) * 2 + 0.001
    prob_npy = prob_npy / prob_npy.sum(axis=1).reshape(prob_npy.shape[0], 1)
    score_npy = get_numpy_rng().rand(batch_size, )
    sample_npy = get_numpy_rng().rand(batch_size, sample_dim) * 1  + 5
    net.forward(is_train=True, prob=prob_npy, mean=mean_npy, var=var_npy)
    net.backward(score=score_npy, policy_backward_action=sample_npy)
    def fd_grad():
        eps = 1E-8
        base_loglikelihood = logmog(prob=prob_npy, mean=mean_npy, var=var_npy, score=score_npy, sample=sample_npy)
        fd_prob_grad = numpy.empty(prob_npy.size, dtype=numpy.float32)
        fd_mean_grad = numpy.empty(mean_npy.size, dtype=numpy.float32)
        fd_var_grad = numpy.empty(var_npy.size, dtype=numpy.float32)
        prob_delta = numpy.zeros(prob_npy.size, dtype=numpy.float32)
        mean_delta = numpy.zeros(mean_npy.size, dtype=numpy.float32)
        var_delta = numpy.zeros(var_npy.size, dtype=numpy.float32)
        for i in range(prob_npy.size):
            prob_delta[i] = eps
            fd_prob_grad[i] = (logmog(prob=prob_npy + prob_delta.reshape(prob_npy.shape), mean=mean_npy, var=var_npy, score=score_npy,
                                     sample=sample_npy) - base_loglikelihood)/eps
            prob_delta[i] = 0
        for i in range(mean_npy.size):
            mean_delta[i] = eps
            fd_mean_grad[i] = (logmog(prob=prob_npy, mean=mean_npy + mean_delta.reshape(mean_npy.shape), var=var_npy,
                                      score=score_npy,
                                      sample=sample_npy) - base_loglikelihood) / eps
            mean_delta[i] = 0
        for i in range(var_npy.size):
            var_delta[i] = eps
            fd_var_grad[i] = (logmog(prob=prob_npy, mean=mean_npy, var=var_npy + var_delta.reshape(var_npy.shape),
                                     score=score_npy,
                                     sample=sample_npy) - base_loglikelihood) / eps
            var_delta[i] = 0
        fd_prob_grad = fd_prob_grad.reshape(prob_npy.shape)
        fd_mean_grad = fd_mean_grad.reshape(mean_npy.shape)
        fd_var_grad = fd_var_grad.reshape(var_npy.shape)
        return fd_prob_grad, fd_mean_grad, fd_var_grad
    fd_prob_grad, fd_mean_grad, fd_var_grad = fd_grad()
    # print 'fd_prob_grad:', fd_prob_grad
    # print 'fd_mean_grad:', fd_mean_grad
    # print 'fd_var_grad:', fd_var_grad
    op_prob_grad = net.executor_pool.inputs_grad_dict.values()[0]['prob'].asnumpy()
    op_mean_grad = net.executor_pool.inputs_grad_dict.values()[0]['mean'].asnumpy()
    op_var_grad = net.executor_pool.inputs_grad_dict.values()[0]['var'].asnumpy()
    # print 'op_prob_grad:', op_prob_grad
    # print 'op_mean_grad:', op_mean_grad
    # print 'op_var_grad:', op_var_grad
    print 'prob_grad_diff:', numpy.square(op_prob_grad - fd_prob_grad).sum()
    print 'mean_grad_diff:', numpy.square(op_mean_grad - fd_mean_grad).sum()
    print 'var_grad_diff:', numpy.square(op_var_grad - fd_var_grad).sum()
#mog_sample_test()
mog_backward_test(batch_size=5, num_centers=11, sample_dim=33)
mog_backward_test(batch_size=4, num_centers=8, sample_dim=33)
mog_backward_test(batch_size=5, num_centers=11, sample_dim=64)
mog_backward_test(batch_size=7, num_centers=5, sample_dim=16*16)





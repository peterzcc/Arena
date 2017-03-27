import numpy as np
import tensorflow as tf
import random
import logging
from collections import OrderedDict
import scipy.optimize

# TODO: review
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32


def numel(x):
    return np.prod(var_shape(x))


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [np.prod(var_shape(v))])
                         for (grad, v) in zip(grads, var_list)])


# set theta
class SetFromFlat(object):
    def __init__(self, var_list, session=None):
        shapes = list(map(var_shape, var_list))
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = tf.placeholder(dtype, [total_size], name="flat_theta")
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(
                    v,
                    tf.reshape(
                        self.theta[start:(start + size)],
                        shape), use_locking=True))
            start += size
        self.op = tf.group(*assigns)
        self.session = session

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


# get theta
class GetFlat(object):
    def __init__(self, var_list, session=None):
        self.op = tf.concat(0, [tf.reshape(v, [numel(v)]) for v in var_list])
        self.session = session

    def __call__(self):
        return self.op.eval(session=self.session)


class EzFlat(object):
    def __init__(self, var_list):
        self.gf = GetFlat(var_list)
        self.sff = SetFromFlat(var_list)

    def set_params_flat(self, theta):
        self.sff(theta)

    def get_params_flat(self):
        return self.gf()


class LbfgsOptimizer(EzFlat):
    def __init__(self, loss, params, symb_args, maxiter=25):
        EzFlat.__init__(self, params)
        self.grad_vector = flatgrad(loss, params)
        self.loss_and_grad = tf.group(*[loss, self.grad_vector])
        # self.f_lossgrad = theano.function(list(symb_args), [loss, flatgrad(loss, params)], **FNOPTS)
        # self.f_losses = theano.function(symb_args, self.all_losses.values(), **FNOPTS)
        self.maxiter = maxiter

    def update(self, session=tf.get_default_session(), **feed):
        """

        Parameters
        ----------
        session: tf.Session
        feed

        Returns
        -------

        """
        thprev = self.get_params_flat()

        def lossandgrad(th):
            self.set_params_flat(th)
            l, g = session.run(self.loss_and_grad, feed_dict=feed)
            g = g.astype('float64')
            return (l, g)

        # losses_before = self.f_losses(*args)
        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=self.maxiter)
        del opt_info['grad']
        logging.debug(opt_info)
        self.set_params_flat(theta)
        # losses_after = self.f_losses(*args)
        info = OrderedDict()
        for (name, lossbefore, lossafter) in zip(self.all_losses.keys(), losses_before, losses_after):
            info[name + "_before"] = lossbefore
            info[name + "_after"] = lossafter
        return info


# TODO: revise this
# def linesearch(f, x, fullstep, expected_improve_rate):
#     accept_ratio = .1
#     max_backtracks = 10
#     fval, old_kl, entropy = f(x)
#     for (_n_backtracks, stepfrac) in enumerate(.3**np.arange(max_backtracks)):
#         xnew = x + stepfrac * fullstep
#         newfval, new_kl, new_ent= f(xnew)
#         # actual_improve = newfval - fval # minimize target object
#         # expected_improve = expected_improve_rate * stepfrac
#         # ratio = actual_improve / expected_improve
#         # if ratio > accept_ratio and actual_improve > 0:
#         #     pms.max_kl *= 1.002
#         #     return xnew
#         if newfval<fval and new_kl<=pms.max_kl:
#             pms.max_kl *=1.002
#             return xnew
#     return x

# def linesearch(f, x, fullstep, expected_improve_rate,accept_ratio=.1,max_backtracks=10,
#                bactrack_factor=.5):
#     fval = f(x)
#     for (_n_backtracks, stepfrac) in enumerate(bactrack_factor**np.arange(max_backtracks)):
#         xnew = x + stepfrac * fullstep
#         newfval = f(xnew)
#         actual_improve = fval - newfval
#         expected_improve = expected_improve_rate * stepfrac
#         ratio = actual_improve / expected_improve
#         if ratio > accept_ratio and actual_improve > 0:
#             return xnew
#     return x

def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    logging.debug("fval before: {}".format(fval))
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):  # 0.5^n
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        logging.debug("a:{}\te:{}\tr:{}".format(actual_improve, expected_improve, ratio))
        if ratio > accept_ratio and actual_improve > 0:
            logging.debug("fval after: {}".format(newfval))
            return xnew
    logging.debug("Failed to find improvement")
    return x


class ZFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


def IDENTITY(x):
    return x


# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape
def cg(f_Ax, b, cg_iters=10, verbose=True, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    print("x.dim: {}".format(x.shape))

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: logging.debug(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        # if callback is not None:
        #     callback(x)
        if verbose: logging.debug(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / (p.dot(z) + 0.0 * 1e-8)  # TODO: reset to nonzero
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / (rdotr + 0.0 * 1e-8)
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    # if callback is not None:
    #     callback(x)
    if verbose: logging.debug(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x

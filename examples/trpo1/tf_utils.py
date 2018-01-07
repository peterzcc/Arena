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
    return tf.concat(axis=0, values=[tf.reshape(grad, [np.prod(var_shape(v))])
                                     for (grad, v) in zip(grads, var_list)])


def flatgrad_batch(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(axis=1, values=[tf.reshape(grad, [var_shape(v)[0], np.prod(var_shape(v)[1:])])
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
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        self.session = session

    def __call__(self):
        return self.op.eval(session=self.session)


class EzFlat(object):
    def __init__(self, var_list, session=None):
        self.gf = GetFlat(var_list, session=session)
        self.sff = SetFromFlat(var_list, session=session)

    def set_params_flat(self, theta):
        self.sff(theta)

    def get_params_flat(self):
        return self.gf()


class LbfgsOptimizer(EzFlat):
    def __init__(self, loss, params, maxiter=25, session=None):
        EzFlat.__init__(self, params, session=session)
        self.grad_vector = flatgrad(loss, params)
        self.loss_and_grad = [loss, self.grad_vector]
        # self.f_lossgrad = theano.function(list(symb_args), [loss, flatgrad(loss, params)], **FNOPTS)
        # self.f_losses = theano.function(symb_args, self.all_losses.values(), **FNOPTS)
        self.maxiter = maxiter
        self.session = session

    def update(self, session=tf.get_default_session(), feed=None):
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
        # for (name, lossbefore, lossafter) in zip(self.all_losses.keys(), losses_before, losses_after):
        #     info[name + "_before"] = lossbefore
        #     info[name + "_after"] = lossafter
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
def aggregate_feature(st, img):
    return st + img  # tf.pad(img, paddings=tf.constant(value=[[0, 0], [0, 2]]))


def concat_without_task(st, img):
    return tf.concat(
        axis=1,
        values=[st[:, 2:], img])

def select_st(st, img):
    final = st + img[0, 0]
    return final
def concat_feature(st, img):
    return tf.concat(
        axis=1,
        values=[st, img])

def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    logging.debug("fval before: {}".format(fval))
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):  # 0.5^n
        diff = stepfrac * fullstep
        xnew = x + diff
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        logging.debug("\nk:{}\ta:{}\te:{}\tr:{}".format(stepfrac, actual_improve, expected_improve, ratio))
        if ratio > accept_ratio and actual_improve > 0:
            logging.debug("fval after: {}".format(newfval))
            return xnew, diff
    logging.debug("Failed to find improvement")
    return x, 0


def run_batched(func, feed, N, session, minibatch_size=64, extra_input={}):
    assert N > 0
    result = None
    for start in range(0, N, minibatch_size):  # TODO: verify this
        end = min(start + minibatch_size, N)
        this_size = end - start
        assert this_size > 0
        slc = range(start, end)
        this_feed = {k: v[slc] for k, v in list(feed.items())}
        this_result = np.array(session.run(func, feed_dict={**this_feed, **extra_input})) * this_size
        if result is None:
            result = this_result
        else:
            result += this_result
    result /= N
    return result


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def stochastic_cg(f_Ax, v, cg_iters=10, verbose=True, residual_tol=1e-10, scale=1.0, initial_guess=None):
    Hjv = v.copy() if initial_guess is None else initial_guess
    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: logging.debug(titlestr % ("iter", "residual norm", "soln norm"))
    if verbose: logging.debug(fmtstr % (0, -1, np.linalg.norm(Hjv)))
    for i in range(cg_iters):
        Hjv_old = Hjv
        new_est = f_Ax(Hjv_old)
        r = v - new_est
        Hjv = Hjv_old + r / scale
        rdotr = r.dot(r)
        if verbose: logging.debug(fmtstr % (i + 1, rdotr, np.linalg.norm(Hjv)))
        if rdotr < residual_tol:
            break
    return Hjv
# http://www.johndcook.com/blog/standard_deviation/
EPS = 1e-6
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
    if verbose: logging.debug(fmtstr % (0, rdotr, np.linalg.norm(x)))
    for i in range(cg_iters):
        # if callback is not None:
        #     callback(x)

        z = f_Ax(p)
        v = rdotr / (p.dot(z) + EPS)
        if i == cg_iters - 1:
            old_x = x
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / (rdotr + EPS)
        p = r + mu * p
        if verbose: logging.debug(fmtstr % (i + 1, newrdotr, np.linalg.norm(x)))
        if i == cg_iters - 1 and newrdotr > rdotr:
            x = old_x
            logging.debug("Last iteration failed, rollback")
        rdotr = newrdotr

        if rdotr < residual_tol:
            break

    # if callback is not None:
    #     callback(x)
    return x

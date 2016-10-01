""" This file defines general utility functions and classes. """
import numpy as np
import scipy as sp
import scipy.ndimage as sp_ndimage
import abc
import logging

LOGGER = logging.getLogger(__name__)

class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable
    namedtuple.
    """
    def __init__(self, variables):
        for var, val in variables.items():
            object.__setattr__(self, var, val)

    # Freeze fields so new ones cannot be set.
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)

"""
Throws a ValueError if value.shape != expected_shape.
Args:
    value: Matrix to shape check.
    expected_shape: A tuple or list of integers.
    name: An optional name to add to the exception message.
"""
def check_shape(value, expected_shape, name=''):

    if value.shape != tuple(expected_shape):
        raise ValueError('Shape mismatch %s: Expected %s, got %s' %
                         (name, str(expected_shape), str(value.shape)))

'''
Compute the LogSumExp function.
'''
def logsum(vec, axis=0, keepdims=True):
    #TODO: Add a docstring.
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec-maxv), axis=axis, keepdims=keepdims)) + maxv

"""
Initial guess at the model using position-velocity assumption.
Note: This code assumes joint positions occupy the first dU state
      indices and joint velocities occupy the next dU.
Args:
    gains: dU dimensional joint gains.
    acc: dU dimensional joint acceleration.
    dX: Dimensionality of the state.
    dU: Dimensionality of the action.
    dt: Length of a time step.
Returns:
    Fd: A dX by dX+dU transition matrix.
    fc: A dX bias vector.
"""
def guess_dynamics(gains, acc, dX, dU, dt):

    #TODO: Use packing instead of assuming which indices are the joint
    #      angles.
    Fd = np.vstack([
        np.hstack([
            np.eye(dU), dt * np.eye(dU), np.zeros((dU, dX - dU*2)),
            dt ** 2 * np.diag(gains)
        ]),
        np.hstack([
            np.zeros((dU, dU)), np.eye(dU), np.zeros((dU, dX - dU*2)),
            dt * np.diag(gains)
        ]),
        np.zeros((dX - dU*2, dX+dU))
    ])
    fc = np.hstack([acc * dt ** 2, acc * dt, np.zeros((dX - dU*2))])
    return Fd, fc

""" Class for performing repeated line searches. """
class LineSearch(object):
    __metaclass__ = abc.ABCMeta

    """
    Construct new line search object, storing necessary data.
    Args:
        min_eta: Minimum value of eta.
    """
    def __init__(self, min_eta):
        self.data = {}
        self.min_eta = min_eta

    """
    Adjust eta using second order bracketed line search.
    Args:
        con: Constraint violation amount.
        new_eta: New values of eta.
        min_eta: Minimum value of eta.
    Returns:
        eta: New found value of eta.
    """
    def bracketing_line_search(self, con, eta, min_eta):
        # Adjust eta.
        if (not self.data or
                (abs(self.data['c1']-con) < 1e-16 * abs(self.data['c1']+con) and
                 abs(self.data['c2']-con) < 1e-16 * abs(self.data['c2']+con))):
            # Take initial step if we don't have multiple points already
            # available.
            self.data = {'c1': con, 'c2': con, 'e1': eta, 'e2': eta}
            if con < 0:  # Too little change.
                rate = abs(1.0 / (eta * con))
                cng = min(max(rate * eta * con, -5.0), 5.0)
                eta = np.exp(np.log(eta) + cng)
            else:  # Too much change.
                rate = 0.01
                eta = eta + con * rate
        else:
            # Choose two points to fit.
            leta = eta
            if (self.data['c1'] <= con and self.data['c2'] > con and
                    self.data['c1'] < 0 and self.data['c2'] > 0):
                mid = 1
                if con < 0:
                    c1 = con
                    c2 = self.data['c2']
                    e1 = leta
                    e2 = self.data['e2']
                else:
                    c1 = self.data['c1']
                    c2 = con
                    e1 = self.data['e1']
                    e2 = leta
            else:
                mid = 0
                if (self.data['c1'] < 0 and self.data['c2'] < 0 and con < 0 and
                        # Too little change in all cases.
                        eta <= self.data['e1'] and eta <= self.data['e2'] and
                        # eta is decreasing.
                        self.data['c1'] != self.data['c2'] and
                        # Change in con is small compared to change in eta.
                        abs(self.data['c2'] - con) / (
                            max(abs(self.data['c2']), abs(con)) *
                            abs(np.log(self.data['e2']) - np.log(eta))
                        ) < 1e-2):

                    # If rate is changing very slowly, try jumping to
                    # lowest possible eta.
                    LOGGER.debug(
                        'Rate is changing slowly, jumping to minimum eta.'
                    )
                    c1 = self.data['c1']
                    c2 = con
                    e1 = self.data['e1']
                    e2 = leta
                    self.data['c1'] = c1
                    self.data['c2'] = c2
                    self.data['e1'] = e1
                    self.data['e2'] = e2
                    eta = max(min_eta, self.min_eta)
                    return eta
                else:
                    if abs(self.data['c1']) <= abs(self.data['c2']):
                        if self.data['c1'] < con:
                            c1 = self.data['c1']
                            c2 = con
                            e1 = self.data['e1']
                            e2 = leta
                        else:
                            c1 = con
                            c2 = self.data['c1']
                            e1 = leta
                            e2 = self.data['e1']
                    else:
                        if self.data['c2'] < con:
                            c1 = self.data['c2']
                            c2 = con
                            e1 = self.data['e2']
                            e2 = leta
                        else:
                            c1 = con
                            c2 = self.data['c2']
                            e1 = leta
                            e2 = self.data['e2']

            # First, try to perform a log-space fit. Note that this may
            # no longer be convex.
            lc1 = c1 * e1
            lc2 = c2 * e2
            le1 = np.log(e1)
            le2 = np.log(e2)

            # Fit quadratic.
            a = (lc2 - lc1) / (2 * (le2 - le1))
            b = 0.5 * (lc1 + lc2 - 2 * a * (le1 + le2))

            # Decide whether we want to solve in the original space
            # instead. Use ratio of gradients. Solve in the original if
            # the ratio of gradients is very high in log space (with 50
            # percent probability).
            if (abs(np.log(abs(lc1 / lc2))) > 2 * abs(np.log(abs(c1 / c2))) and
                    c1 * c2 < 0 and np.random.rand() < 0.5):
                solve_orig = 1
            else:
                solve_orig = 0

            # Compute minimium.
            if a < 0 and not solve_orig:  # Concave, good to go.
                nlogeta = -b / (2 * a)
            else:  # Solve in original space.
                if solve_orig:
                    LOGGER.debug(
                        'Solving non-log problem due to ratio choice: %f %f',
                        abs(np.log(abs(lc1 / lc2))), abs(np.log(abs(c1 / c2)))
                    )
                if not solve_orig:
                    LOGGER.debug(
                        'Solving non-log problem due to a > 0: %f %f',
                        abs(np.log(abs(lc1 / lc2))), abs(np.log(abs(c1 / c2)))
                    )

                # Fit quadratic.
                a = (c2 - c1) / (2 * (e2 - e1))
                b = 0.5 * (c1 + c2 - 2 * a * (e1 + e2))
                if a < 0:  # Concave, good to go.
                    nlogeta = np.log(max(-b / (2 * a), 1e-50))
                else:
                    # Something is very wrong.
                    LOGGER.warning('Dual is not concave!')
                    rate = abs(1.0 / (con * eta))
                    cng = min(max(rate * con * eta, -5), 5)
                    nlogeta = np.log(eta) + cng

            LOGGER.debug('Bracket points: %f %f (%f %f) a=%f b=%f',
                         e1, e2, c1, c2, a, b)

            # Bound change.
            if mid == 1:
                MAX_LOG_STEP = np.Inf
            else:
                MAX_LOG_STEP = 4

            if nlogeta > np.log(eta):
                nlogeta = np.log(eta) + min(nlogeta-np.log(eta), MAX_LOG_STEP)
            else:
                nlogeta = np.log(eta) + max(nlogeta-np.log(eta), -MAX_LOG_STEP)

            # Save old info and set new eta.
            self.data['c1'] = c1
            self.data['c2'] = c2
            self.data['e1'] = e1
            self.data['e2'] = e2
            eta = max(np.exp(nlogeta), max(min_eta, self.min_eta))
        return eta

def traj_distr_kl(new_mu, new_sigma, new_traj_distr, prev_traj_distr):
    """
    Compute KL divergence between new and previous trajectory
    distributions.
    Args:
        new_mu: T x dX, mean of new trajectory distribution.
        new_sigma: T x dX x dX, variance of new trajectory distribution.
        new_traj_distr: A linear Gaussian policy object, new
            distribution.
        prev_traj_distr: A linear Gaussian policy object, previous
            distribution.
    Returns:
        kl_div: The KL divergence between the new and previous
            trajectories.
    """
    # Constants.
    T = new_mu.shape[0]
    dU = new_traj_distr.dU

    # Initialize vector of divergences for each time step.
    kl_div = np.zeros(T)

    # Step through trajectory.
    for t in range(T):
        # Fetch matrices and vectors from trajectory distributions.
        mu_t = new_mu[t, :]
        sigma_t = new_sigma[t, :, :]
        K_prev = prev_traj_distr.K[t, :, :]
        K_new = new_traj_distr.K[t, :, :]
        k_prev = prev_traj_distr.k[t, :]
        k_new = new_traj_distr.k[t, :]
        chol_prev = prev_traj_distr.chol_pol_covar[t, :, :]
        chol_new = new_traj_distr.chol_pol_covar[t, :, :]

        # Compute log determinants and precision matrices.
        logdet_prev = 2 * sum(np.log(np.diag(chol_prev)))
        logdet_new = 2 * sum(np.log(np.diag(chol_new)))
        prc_prev = sp.linalg.solve_triangular(
            chol_prev, sp.linalg.solve_triangular(chol_prev.T, np.eye(dU),
                                                  lower=True)
        )
        prc_new = sp.linalg.solve_triangular(
            chol_new, sp.linalg.solve_triangular(chol_new.T, np.eye(dU),
                                                 lower=True)
        )

        # Construct matrix, vector, and constants.
        M_prev = np.r_[
            np.c_[K_prev.T.dot(prc_prev).dot(K_prev), -K_prev.T.dot(prc_prev)],
            np.c_[-prc_prev.dot(K_prev), prc_prev]
        ]
        M_new = np.r_[
            np.c_[K_new.T.dot(prc_new).dot(K_new), -K_new.T.dot(prc_new)],
            np.c_[-prc_new.dot(K_new), prc_new]
        ]
        v_prev = np.r_[K_prev.T.dot(prc_prev).dot(k_prev),
                       -prc_prev.dot(k_prev)]
        v_new = np.r_[K_new.T.dot(prc_new).dot(k_new), -prc_new.dot(k_new)]
        c_prev = 0.5 * k_prev.T.dot(prc_prev).dot(k_prev)
        c_new = 0.5 * k_new.T.dot(prc_new).dot(k_new)

        # Compute KL divergence at timestep t.
        kl_div[t] = max(
            0,
            -0.5 * mu_t.T.dot(M_new - M_prev).dot(mu_t) -
            mu_t.T.dot(v_new - v_prev) - c_new + c_prev -
            0.5 * np.sum(sigma_t * (M_new-M_prev)) - 0.5 * logdet_new +
            0.5 * logdet_prev
        )

    # Add up divergences across time to get total divergence.
    return np.sum(kl_div)

def generate_noise(T, dU, std, smooth, var, renorm):
    """
    Generate a T x dU gaussian-distributed noise vector. This will
    approximately have mean 0 and variance 1, ignoring smoothing.

    Args:
        T: Number of time steps.
        dU: Dimensionality of actions.
    Hyperparams:
        smooth: Whether or not to perform smoothing of noise.
        var : If smooth=True, applies a Gaussian filter with this
            variance.
        renorm : If smooth=True, renormalizes data to have variance 1
            after smoothing.
    """
    # smooth, var = hyperparams['smooth_noise'], hyperparams['smooth_noise_var']
    # renorm = hyperparams['smooth_noise_renormalize']
    noise = np.random.randn(T, dU) * std
    if smooth:
        # Smooth noise. This violates the controller assumption, but
        # might produce smoother motions.
        for i in range(dU):
            noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
        if renorm:
            variance = np.var(noise, axis=0)
            noise = noise / np.sqrt(variance)
    return noise
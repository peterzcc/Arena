""" This file defines several classes for dynamics estimation"""
import abc
import logging

import numpy as np

from arena.advanced.gps.prior import DynamicsPriorGMM

LOGGER = logging.getLogger(__name__)


class Dynamics(object):
    """ Dynamics superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        # Fitted dynamics: x_t+1 = Fm * [x_t;u_t] + fv.
        self.Fm = np.array(np.nan)
        self.fv = np.array(np.nan)
        self.dyn_covar = np.array(np.nan)  # Covariance.

    @abc.abstractmethod
    def update_prior(self, X, U):
        """
        Update dynamics prior.
        X, state data, N x T x dX
        U, action data, N x T x dU
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def get_prior(self):
        """ Returns prior object. """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def fit(self, X, U):
        """
        Fit dynamics.
        X, state data, N x T x dX
        U, action data, N x T x dU
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def copy(self):
        """ Return a copy of the dynamics estimate. """
        dyn = type(self)(self._hyperparams)
        dyn.Fm = np.copy(self.Fm)
        dyn.fv = np.copy(self.fv)
        dyn.dyn_covar = np.copy(self.dyn_covar)
        return dyn


class DynamicsLR(Dynamics):
    """ Dynamics with linear regression, with constant prior. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None

    def update_prior(self, X, U):
        """ Update dynamics prior. """
        # Nothing to do - constant prior.
        pass

    def get_prior(self):
        """ Return the dynamics prior, or None if constant prior. """
        return None

    def fit(self, X, U):
        """ Fit dynamics. """
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics wih least squares regression.
        for t in range(T - 1):
            xux = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]
            xux_mean = np.mean(xux, axis=0)
            empsig = (xux - xux_mean).T.dot(xux - xux_mean) / (N - 1)
            sigma = 0.5 * (empsig + empsig.T)
            sigma[it, it] += self._hyperparams['regularization'] * np.eye(dX+dU)

            Fm = np.linalg.pinv(sigma[it, it]).dot(sigma[it, ip]).T
            fv = xux_mean[ip] - Fm.dot(xux_mean[it])

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv

            dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)
            self.dyn_covar[t, :, :] = 0.5 * (dyn_covar + dyn_covar.T)


class DynamicsLRPrior(Dynamics):
    """ Dynamics with linear regression, with GMM prior. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        # self.prior = DynamicsPriorGMM(self._hyperparams)
        self.prior = \
                DynamicsPriorGMM(self._hyperparams['prior'])

    def update_prior(self, X, U):
        """ Update dynamics prior. """
        self.prior.update(X, U)

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior

    #TODO: Merge this with DynamicsLR.fit - lots of duplicated code.
    def fit(self, X, U):
        """ Fit dynamics. """
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics with least squares regression.
        for t in range(T - 1):
            xux = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]

            mu0, Phi, m, n0 = self.prior.eval(dX, dU, xux)

            xux_mean = np.mean(xux, axis=0)
            empsig = (xux - xux_mean).T.dot(xux - xux_mean) / (N - 1)
            empsig = 0.5 * (empsig + empsig.T)

            sigma = (N * empsig + Phi + (N * m) / (N + m) *
                     np.outer(xux_mean - mu0, xux_mean - mu0)) / (N + n0)
            sigma = 0.5 * (sigma + sigma.T)
            sigma[it, it] += self._hyperparams['regularization'] * np.eye(dX+dU)

            Fm = np.linalg.pinv(sigma[it, it]).dot(sigma[it, ip]).T
            fv = xux_mean[ip] - Fm.dot(xux_mean[it])

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv

            dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)
            self.dyn_covar[t, :, :] = 0.5 * (dyn_covar + dyn_covar.T)
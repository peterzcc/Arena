""" This file defines a Gaussian mixture model class. """
import logging
import copy

import numpy as np
import scipy.linalg

from arena.advanced.gps.utils import logsum
from arena.advanced.gps.config import DYN_PRIOR_GMM

LOGGER = logging.getLogger(__name__)


""" Gaussian Mixture Model. """
class GMM(object):
    def __init__(self, init_sequential=False, eigreg=False, warmstart=True):
        self.init_sequential = init_sequential
        self.eigreg = eigreg
        self.warmstart = warmstart
        self.sigma = None

    """
    Evaluate dynamics prior.
    Args:
        pts: A N x D array of points.
    """
    def inference(self, pts):

        # Compute posterior cluster weights.
        logwts = self.clusterwts(pts)

        # Compute posterior mean and covariance.
        mu0, Phi = self.moments(logwts)

        # Set hyperparameters.
        m = self.N
        n0 = m - 2 - mu0.shape[0]

        # Normalize.
        m = float(m) / self.N
        n0 = float(n0) / self.N
        return mu0, Phi, m, n0

    """
    Compute log observation probabilities under GMM.
    Args:
        data: A N x D array of points.
    Returns:
        logobs: A N x K array of log probabilities (for each point
            on each cluster).
    """
    def estep(self, data):
        # Constants.
        K = self.sigma.shape[0]
        Di = data.shape[1]
        N = data.shape[0]

        # Compute probabilities.
        data = data.T
        mu = self.mu[:, 0:Di].T
        mu_expand = np.expand_dims(np.expand_dims(mu, axis=1), axis=1)
        assert mu_expand.shape == (Di, 1, 1, K)
        # Calculate for each point distance to each cluster.
        data_expand = np.tile(data, [K, 1, 1, 1]).transpose([2, 3, 1, 0])
        diff = data_expand - np.tile(mu_expand, [1, N, 1, 1])
        assert diff.shape == (Di, N, 1, K)
        Pdiff = np.zeros_like(diff)
        cconst = np.zeros((1, 1, 1, K))

        for i in range(K):
            U = scipy.linalg.cholesky(self.sigma[i, :Di, :Di],
                                      check_finite=False)
            Pdiff[:, :, 0, i] = scipy.linalg.solve_triangular(
                U, scipy.linalg.solve_triangular(
                    U.T, diff[:, :, 0, i], lower=True, check_finite=False
                ), check_finite=False
            )
            cconst[0, 0, 0, i] = -np.sum(np.log(np.diag(U))) - 0.5 * Di * \
                    np.log(2 * np.pi)

        logobs = -0.5 * np.sum(diff * Pdiff, axis=0, keepdims=True) + cconst
        assert logobs.shape == (1, N, 1, K)
        logobs = logobs[0, :, 0, :] + self.logmass.T
        return logobs

    """
    Compute the moments of the cluster mixture with logwts.
    Args:
        logwts: A K x 1 array of log cluster probabilities.
    Returns:
        mu: A (D,) mean vector.
        sigma: A D x D covariance matrix.
    """
    def moments(self, logwts):
        # Exponentiate.
        wts = np.exp(logwts)

        # Compute overall mean.
        mu = np.sum(self.mu * wts, axis=0)

        # Compute overall covariance.
        # For some reason this version works way better than the "right"
        # one... could we be computing xxt wrong?
        diff = self.mu - np.expand_dims(mu, axis=0)
        diff_expand = np.expand_dims(diff, axis=1) * \
                np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(wts, axis=2)
        sigma = np.sum((self.sigma + diff_expand) * wts_expand, axis=0)
        return mu, sigma

    """
    Compute cluster weights for specified points under GMM.
    Args:
        data: An N x D array of points
    Returns:
        A K x 1 array of average cluster log probabilities.
    """
    def clusterwts(self, data):
        # Compute probability of each point under each cluster.
        logobs = self.estep(data)

        # Renormalize to get cluster weights.
        logwts = logobs - logsum(logobs, axis=1)

        # Average the cluster probabilities.
        logwts = logsum(logwts, axis=0) - np.log(data.shape[0])
        return logwts.T

    """
    Run EM to update clusters.
    Args:
        data: An N x D data matrix, where N = number of data points.
        K: Number of clusters to use.
    """
    def update(self, data, K, max_iterations=100):
        # Constants.
        N = data.shape[0]
        Do = data.shape[1]

        LOGGER.debug('Fitting GMM with %d clusters on %d points', K, N)

        if (not self.warmstart or self.sigma is None or
                K != self.sigma.shape[0]):
            # Initialization.
            LOGGER.debug('Initializing GMM.')
            self.sigma = np.zeros((K, Do, Do))
            self.mu = np.zeros((K, Do))
            self.logmass = np.log(1.0 / K) * np.ones((K, 1))
            self.mass = (1.0 / K) * np.ones((K, 1))
            self.N = data.shape[0]
            N = self.N

            # Set initial cluster indices.
            if not self.init_sequential:
                cidx = np.random.randint(0, K, size=(1, N))
            else:
                raise NotImplementedError()

            # Initialize.
            for i in range(K):
                cluster_idx = (cidx == i)[0]
                mu = np.mean(data[cluster_idx, :], axis=0)
                diff = (data[cluster_idx, :] - mu).T
                sigma = (1.0 / K) * (diff.dot(diff.T))
                self.mu[i, :] = mu
                self.sigma[i, :, :] = sigma + np.eye(Do) * 2e-6

        prevll = -float('inf')
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logobs = self.estep(data)

            # Compute log-likelihood.
            ll = np.sum(logsum(logobs, axis=1))
            LOGGER.debug('GMM itr %d/%d. Log likelihood: %f',
                         itr, max_iterations, ll)
            if ll < prevll:
                # TODO: Why does log-likelihood decrease sometimes?
                LOGGER.debug('Log-likelihood decreased! Ending on itr=%d/%d',
                             itr, max_iterations)
                break
            if np.abs(ll-prevll) < 1e-2:
                LOGGER.debug('GMM converged on itr=%d/%d',
                             itr, max_iterations)
                break
            prevll = ll

            # Renormalize to get cluster weights.
            logw = logobs - logsum(logobs, axis=1)
            assert logw.shape == (N, K)

            # Renormalize again to get weights for refitting clusters.
            logwn = logw - logsum(logw, axis=0)
            assert logwn.shape == (N, K)
            w = np.exp(logwn)

            # M-step: update clusters.
            # Fit cluster mass.
            self.logmass = logsum(logw, axis=0).T
            self.logmass = self.logmass - logsum(self.logmass, axis=0)
            assert self.logmass.shape == (K, 1)
            self.mass = np.exp(self.logmass)
            # Reboot small clusters.
            w[:, (self.mass < (1.0 / K) * 1e-4)[:, 0]] = 1.0 / N
            # Fit cluster means.
            w_expand = np.expand_dims(w, axis=2)
            data_expand = np.expand_dims(data, axis=1)
            self.mu = np.sum(w_expand * data_expand, axis=0)
            # Fit covariances.
            wdata = data_expand * np.sqrt(w_expand)
            assert wdata.shape == (N, K, Do)
            for i in range(K):
                # Compute weighted outer product.
                XX = wdata[:, i, :].T.dot(wdata[:, i, :])
                mu = self.mu[i, :]
                self.sigma[i, :, :] = XX - np.outer(mu, mu)

                if self.eigreg:  # Use eigenvalue regularization.
                    raise NotImplementedError()
                else:  # Use quick and dirty regularization.
                    sigma = self.sigma[i, :, :]
                    self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + \
                            1e-6 * np.eye(Do)


"""
A dynamics prior encoded as a GMM over [x_t, u_t, x_t+1] points.
See:
    S. Levine*, C. Finn*, T. Darrell, P. Abbeel, "End-to-end
    training of Deep Visuomotor Policies", arXiv:1504.00702,
    Appendix A.3.
"""
class DynamicsPriorGMM(object):
    """
    Hyperparameters:
        min_samples_per_cluster: Minimum samples per cluster.
        max_clusters: Maximum number of clusters to fit.
        max_samples: Maximum number of trajectories to use for
            fitting the GMM at any given time.
        strength: Adjusts the strength of the prior.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(DYN_PRIOR_GMM)
        config.update(hyperparams)
        self._hyperparams = config
        self.X = None
        self.U = None
        self.gmm = GMM()
        self._min_samp = self._hyperparams['min_samples_per_cluster']
        self._max_samples = self._hyperparams['max_samples']
        self._max_clusters = self._hyperparams['max_clusters']
        self._strength = self._hyperparams['strength']

    """ Return dynamics prior for initial time step. """
    def initial_state(self):
        # Compute mean and covariance.
        mu0 = np.mean(self.X[:, 0, :], axis=0)
        Phi = np.diag(np.var(self.X[:, 0, :], axis=0))

        # Factor in multiplier.
        n0 = self.X.shape[2] * self._strength
        m = self.X.shape[2] * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi = Phi * m
        return mu0, Phi, m, n0

    """
    Update prior with additional data.
    Args:
        X: A N x T x dX matrix of sequential state data.
        U: A N x T x dU matrix of sequential control data.
    """
    def update(self, X, U):
        # Constants.
        T = X.shape[1] - 1

        # Append data to dataset.
        if self.X is None:
            self.X = X
        else:
            self.X = np.concatenate([self.X, X], axis=0)

        if self.U is None:
            self.U = U
        else:
            self.U = np.concatenate([self.U, U], axis=0)

        # Remove excess samples from dataset.
        start = max(0, self.X.shape[0] - self._max_samples + 1)
        self.X = self.X[start:, :]
        self.U = self.U[start:, :]

        # Compute cluster dimensionality.
        Do = X.shape[2] + U.shape[2] + X.shape[2]  #TODO: Use Xtgt.

        # Create dataset.
        N = self.X.shape[0]
        xux = np.reshape(
            np.c_[self.X[:, :T, :], self.U[:, :T, :], self.X[:, 1:(T+1), :]],
            [T * N, Do]
        )

        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samp))))
        LOGGER.debug('Generating %d clusters for dynamics GMM.', K)

        # Update GMM.
        self.gmm.update(xux, K)

    """
    Evaluate prior.
    Args:
        pts: A N x Dx+Du+Dx matrix.
    """
    def eval(self, Dx, Du, pts):
        # Construct query data point by rearranging entries and adding
        # in reference.
        assert pts.shape[1] == Dx + Du + Dx

        # Perform query and fix mean.
        mu0, Phi, m, n0 = self.gmm.inference(pts)

        # Factor in multiplier.
        n0 = n0 * self._strength
        m = m * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0
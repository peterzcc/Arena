""" This file defines the base algorithm class. """

import abc
import copy
import logging

import numpy as np

from arena.advanced.gps.traj_opt import TrajectoryInfo, TrajOptLQRPython
from arena.advanced.gps.dynamics import DynamicsLRPrior
from arena.advanced.gps.config import ALG

LOGGER = logging.getLogger(__name__)


class Optimizer(object):
    """ Optimizer superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams={}):
        config = copy.deepcopy(ALG)
        config.update(hyperparams)
        self._hyperparams = config

        self.iteration_count = 0
        self.prev_traj_info = None
        self.prev_policy = None
        self.base_kl_step = self._hyperparams['kl_step']

    @abc.abstractmethod
    def iteration(self, policy, traj_info, state_samples, action_samples):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")

    def _update_dynamics(self, traj_info, state_samples, action_samples):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
        """
        if self.iteration_count >= 1:
            # save the last step dynamics for kl divergence checking
            self.prev_traj_info.dynamics = traj_info.dynamics.copy()

        traj_info.dynamics.update_prior(state_samples, action_samples)

        traj_info.dynamics.fit(state_samples, action_samples)

        init_X = state_samples[:, 0, :]
        x0mu = np.mean(init_X, axis=0)
        traj_info.x0mu = x0mu
        traj_info.x0sigma = np.diag(
            np.maximum(np.var(init_X, axis=0),
                       self._hyperparams['initial_state_var'])
        )

        prior = traj_info.dynamics.get_prior()
        if prior:
            mu0, Phi, priorm, n0 = prior.initial_state()
            N = state_samples.shape[0]
            traj_info.x0sigma += \
                    Phi + (N*priorm) / (N+priorm) * \
                    np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)


class TrajOptOptimizer(Optimizer):
    """ Sample-based trajectory optimization. """
    def __init__(self, hyperparams={}):
        Optimizer.__init__(self, hyperparams)
        self.traj_opt = TrajOptLQRPython(hyperparams)
        self.step_mult = 1.0
        self.eta = 1.0

    def iteration(self, policy, traj_info, state_samples, action_samples):
        '''
        Run iteration of LQR.
        Parameters
        ----------
        policy
        state_samples
        action_samples

        Returns
        -------

        '''

        # Update dynamics model using all samples.
        self._update_dynamics(traj_info, state_samples, action_samples)

        self._update_step_size(policy, traj_info)  # KL Divergence step size.

        # Run inner loop to compute new policies.
        for _ in range(self._hyperparams['inner_iterations']):
            new_policy = self._update_trajectories(policy, traj_info)

        # self._advance_iteration_variables()
        self.iteration_count += 1
        self.prev_traj_info = copy.deepcopy(traj_info)
        self.prev_policy = copy.deepcopy(policy)
        policy.set_params(new_policy)

    def _update_step_size(self, policy, traj_info):
        """ Evaluate costs on samples, and adjust the step size. """
        # Adjust step size relative to the previous iteration.
        if self.iteration_count >= 1:
            # self._stepadjust(m)

            # Compute values under Laplace approximation. This is the policy
            # that the previous samples were actually drawn from under the
            # dynamics that were estimated from the previous samples.
            previous_laplace_obj = self.traj_opt.estimate_cost(
                self.prev_policy, self.prev_traj_info
            )
            # This is the policy that we just used under the dynamics that
            # were estimated from the previous samples (so this is the cost
            # we thought we would have).
            new_predicted_laplace_obj = self.traj_opt.estimate_cost(
                policy, self.prev_traj_info
            )

            # This is the actual cost we have under the current trajectory
            # based on the latest samples.
            new_actual_laplace_obj = self.traj_opt.estimate_cost(
                policy, traj_info
            )

            # Measure the entropy of the current trajectory (for printout).
            ent = 0
            T = traj_info.Cm.shape[0]
            for t in range(T):
                ent = ent + np.sum(
                    np.log(np.diag(policy.chol_pol_covar[t, :, :]))
                )

            # Compute actual objective values based on the samples.
            previous_mc_obj = np.mean(np.sum(self.prev_traj_info.cs, axis=1), axis=0)
            new_mc_obj = np.mean(np.sum(traj_info.cs, axis=1), axis=0)

            LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f',
                         ent, previous_mc_obj, new_mc_obj)

            # Compute predicted and actual improvement.
            predicted_impr = np.sum(previous_laplace_obj) - \
                             np.sum(new_predicted_laplace_obj)
            actual_impr = np.sum(previous_laplace_obj) - \
                          np.sum(new_actual_laplace_obj)

            # Print improvement details.
            LOGGER.debug('Previous cost: Laplace: %f MC: %f',
                         np.sum(previous_laplace_obj), previous_mc_obj)
            LOGGER.debug('Predicted new cost: Laplace: %f MC: %f',
                         np.sum(new_predicted_laplace_obj), new_mc_obj)
            LOGGER.debug('Actual new cost: Laplace: %f MC: %f',
                         np.sum(new_actual_laplace_obj), new_mc_obj)
            LOGGER.debug('Predicted/actual improvement: %f / %f',
                         predicted_impr, actual_impr)

            # self._set_new_mult(predicted_impr, actual_impr, m)
            # Model improvement as I = predicted_dI * KL + penalty * KL^2,
            # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
            # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
            # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
            # Therefore, the new multiplier is given by pred/2*(pred-act).
            new_mult = predicted_impr / (2.0 * max(1e-4,
                                                   predicted_impr - actual_impr))
            new_mult = max(0.1, min(5.0, new_mult))
            new_step = max(
                min(new_mult * self.step_mult,
                    self._hyperparams['max_step_mult']),
                self._hyperparams['min_step_mult']
            )
            self.step_mult = new_step

            if new_mult > 1:
                LOGGER.debug('Increasing step size multiplier to %f', new_step)
            else:
                LOGGER.debug('Decreasing step size multiplier to %f', new_step)

    def _update_trajectories(self, policy, traj_info):
        """
        Compute new linear Gaussian controllers.
        """
        new_policy, eta \
            = self.traj_opt.update(traj_info, policy, self.step_mult, self.eta, self.base_kl_step)
        self.eta = eta
        return new_policy


    def compute_costs(self, m, eta):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        fCm, fcv = traj_info.Cm / eta, traj_info.cv / eta
        K, ipc, k = traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k

        # Add in the trajectory divergence term.
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] += np.vstack([
                np.hstack([
                    K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                    -K[t, :, :].T.dot(ipc[t, :, :])
                ]),
                np.hstack([
                    -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                ])
            ])
            fcv[t, :] += np.hstack([
                K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),
                -ipc[t, :, :].dot(k[t, :])
            ])

        return fCm, fcv
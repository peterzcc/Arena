''' Default configuration and hyperparameter values '''

'''
------------------------------
Dynamics related parameters.
------------------------------
'''
# DynamicsPriorGMM
DYN_PRIOR_GMM = {
    'min_samples_per_cluster': 20,
    'max_clusters': 50,
    'max_samples': 20,
    'strength': 1.0,
}


'''
------------------------------
Policy related parameters.
------------------------------
'''
# Initial Linear Gaussian Trajectory Distributions, PD-based initializer.
INIT_LG_PD = {
    'init_var': 10.0,
    'pos_gains': 10.0, # position gains
    'vel_gains_mult': 0.01,  # velocity gains multiplier on pos_gains
    'init_action_offset': None,
}

# Initial Linear Gaussian Trajectory distribution, LQR-based initializer.
INIT_LG_LQR = {
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'final_weight': 1.0,
    # Parameters for guessing dynamics
    'init_acc': [],  # dU vector of accelerations, default zeros.
    'init_gains': [],  # dU vector of gains, default ones.
}

# PolicyPrior
POLICY_PRIOR = {
    'strength': 1e-4,
}

# PolicyPriorGMM
POLICY_PRIOR_GMM = {
    'min_samples_per_cluster': 20,
    'max_clusters': 50,
    'max_samples': 20,
    'strength': 1.0,
    'keep_samples': True,
}


'''
------------------------------
Trajectory optimization related parameters.
------------------------------
'''
# TrajOptLQR
TRAJ_OPT_LQR = {
    # Dual variable updates for non-PD Q-function.
    'del0': 1e-4,
    'eta_error_threshold': 1e16,
    'min_eta': 1e-4,
    # Constants used in TrajOptLQR.
    'DGD_MAX_ITER': 50,
    'THRESHA': 1e-4,  # First convergence threshold.
    'THRESHB': 1e-3,  # Second convergence threshold.
}

'''
------------------------------
Algorithm related parameters.
------------------------------
'''
# Algorithm
ALG = {
    'inner_iterations': 1,  # Number of iterations.
    'min_eta': 1e-5,  # Minimum initial lagrange multiplier in DGD for
                      # trajectory optimization.
    'kl_step':0.2,
    'min_step_mult':0.01,
    'max_step_mult':10.0,
    'sample_decrease_var':0.5,
    'sample_increase_var':1.0,
    # Trajectory settings.
    'initial_state_var':1e-6,
    'init_traj_distr': None,  # A list of initial LinearGaussianPolicy
                              # objects for each condition.
    # Trajectory optimization.
    'traj_opt': None,
    # Dynamics hyperaparams.
    'dynamics': None,
    # Costs.
    'cost': None,  # A list of Cost objects for each condition.
    # Whether or not to sample with neural net policy (only for badmm/mdgps).
    'sample_on_policy': False,
}
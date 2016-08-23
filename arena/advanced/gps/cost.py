""" This file defines various costs related functions """
import abc
import copy

import numpy as np


def evall1l2term(wp, d, Jd, Jdd, l1, l2, alpha=1e-2):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.
    loss = (0.5 * l2 * d^2) + (l1 * sqrt(alpha + d^2))
    Args:
        wp: T x D matrix with weights for each dimension and time step.
        d: T x D states to evaluate norm on.
        Jd: T x D x Dx Jacobian - derivative of d with respect to state.
        Jdd: T x D x Dx x Dx Jacobian - 2nd derivative of d with respect
            to state.
        l1: l1 loss weight.
        l2: l2 loss weight.
        alpha: Constant added in square root.
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 + \
            np.sqrt(alpha + np.sum(dscl ** 2, axis=1)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (
        dscls / np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1
    )
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims(
        np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1
    )
    d2 = l1 * (
        (np.expand_dims(np.eye(wp.shape[1]), axis=0) *
         (np.expand_dims(wp ** 2, axis=1) / psq)) -
        ((np.expand_dims(dscls, axis=1) *
          np.expand_dims(dscls, axis=2)) / psq ** 3)
    )
    d2 += l2 * (
        np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1])
    )

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum(Jd_expand_1 * Jd_expand_2 * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0, 2, 1])

    return l, lx, lxx





def cost_action(sample_x, sample_u, tgt_u, wp):
    '''
    Evaluate action cost function and derivatives on a sample
    Parameters
    ----------
    sample_x: state sample
    sample_u: action sample
    tgt_u: action targets
    wp: weights multiplier

    Returns
    -------
    l, lx, lu, lxx, luu, lux
    '''
    T = sample_x.shape[0]
    Dx = sample_x.shape[1]
    Du = sample_u.shape[1]
    l = 0.5 * np.sum(wp * ((sample_u - tgt_u)**2), axis=1)
    lu = wp * (sample_u - tgt_u)
    lx = np.zeros((T, Dx))
    luu = np.tile(np.diag(wp), [T, 1, 1])
    lxx = np.zeros((T, Dx, Dx))
    lux = np.zeros((T, Du, Dx))
    return l, lx, lu, lxx, luu, lux


def cost_state(sample_x, sample_u, tgt_xs, state_indices, wps, l2s, l1s):
    '''
    Evaluate state cost function and derivatives on a sample
    Parameters
    ----------
    sample_x: state sample
    sample_u: action sample
    tgt_xs: a list of state targets
    state_indices: a list of state indices used for cost computation, [[1,3], [3,4], ...].
    wps: a list of weight multipliers used for cost computation
    l2s: a list of l2 parameters
    l1s: a list of l1 parameters

    Returns
    -------
    l, lx, lu, lxx, luu, lux
    '''
    T = sample_x.shape[0]
    Dx = sample_x.shape[1]
    Du = sample_u.shape[1]

    final_l = np.zeros(T)
    final_lu = np.zeros((T, Du))
    final_lx = np.zeros((T, Dx))
    final_luu = np.zeros((T, Du, Du))
    final_lxx = np.zeros((T, Dx, Dx))
    final_lux = np.zeros((T, Du, Dx))

    for i in range(len(state_indices)):
        x_ind = slice(state_indices[i][0], state_indices[i][1])
        x = sample_x[:, x_ind]
        dim_x = x.shape[1]
        tgt = tgt_xs[i]
        wp = np.tile(wps[i], [T, 1])
        l2 = l2s[i]
        l1 = l1s[i]
        dist = x - tgt
        l, ls, lss = evall1l2term(wp, dist, np.tile(np.eye(dim_x), [T, 1, 1]),
                                  np.zeros((T, dim_x, dim_x, dim_x)),
                                  l1, l2)
        final_l += l
        final_lx[:, x_ind] = ls
        final_lxx[:, x_ind, x_ind] = lss

    return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux


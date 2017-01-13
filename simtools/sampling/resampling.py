#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""Resample an existing ensemble, based on an array of weights.
Optionally, a scaled version of the weights may be used, with 
addition of noise, according to Annan and Hargreave's Iterative Importance Sampling.

References
----------
Annan, J. D., & Hargreaves, J. C. (2010). Efficient identification of 
ocean thermodynamics in a physical/biogeochemical ocean model with an iterative 
Importance Sampling method. Ocean Modelling, 32(3-4), 205-215. 
doi:10.1016/j.ocemod.2010.02.003

Douc and Cappe. 2005. Comparison of resampling schemes for particle filtering.
ISPA2005, Proceedings of the 4th Symposium on Image and Signal Processing.

Hol, Jeroen D., Thomas B. Schön, and Fredrik Gustafsson, 
"On Resampling Algorithms for Particle Filters", 
in NSSPW - Nonlinear Statistical Signal Processing Workshop 2006, 
2006 <http://dx.doi.org/10.1109/NSSPW.2006.4378824>
"""
from __future__ import division, print_function
import logging
import numpy as np


RESAMPLING_METHOD = "residual"
EPSILON = 0.05  # start value for adaptive_posterior_exponent (kept if NEFF in NEFF_BOUNDS)
EPSILON_BOUNDS = (1e-3, 0.1)  # has priority over NEFF_BOUNDS
NEFF_BOUNDS = (0.5, 0.9)
DEFAULT_SIZE = 500
DEFAULT_ALPHA_TARGET = 0.95



def _get_Neff(weights, normalize=True):
    """ Return an estimate of the effective ensemble size
    """
    if normalize:
        weightssum = np.sum(weights)
        weights = weights/weightssum
    Neff = 1./np.sum(weights**2)
    return Neff


def adaptive_posterior_exponent(likelihood, epsilon=None, neff_bounds=NEFF_BOUNDS):
    """ Compute likelihood exponents to avoid ensemble collapse

    Resampling weights are computed as:

        weights ~ likelihood ** epsilon

    where epsilon is an exponent (between 0 and 1) chosen so that the effective
    ensemble size of the resampled ensemble remains reasonable, thereby 
    avoiding ensemble collapse (where only very few of the original members 
    are resampled, due to large differences in the likelihood).

    If epsilon is not provided, it will be estimated dynamically to yields 
    an effective ensemble size between 0.5 and 0.9 of the original ensemble.

    Parameters
    ----------
    likelihood
    epsilon : initial value for epsilon
    neff_bounds : acceptable effective ensemble ratio

    Returns
    -------
    epsilon : exponent such that weights = likelihood**epsilon

    Notes
    -----
    Small epsilon value means flatter likelihood, more homogeneous resampling.
    and vice versa for large epsilon value. 

    References
    ----------
    Annan and Hargreaves, 2010, Ocean Modelling
    """
    # compute appropriate weights
    if np.sum(likelihood) == 0:
        raise RuntimeError('No ensemble member has a likelihood greater than zero: consider using less constraints')
    N = np.size(likelihood)

    # CHECK FOR CONVERGENCE: effective ensemble size of the model is equal to 90% of that of a uniform distribution
    Neff_weighted_obs = _get_Neff(likelihood)
    ratio_prior = Neff_weighted_obs/N

    logging.info("Epsilon tuning:")
    logging.info("...no epsilon (eps=1): Neff/N = {}".format(ratio_prior))

    # Now adjust the likelihood function so as to have an effective size 
    # between 50% and 90% that of the previous ensemble (that is, because of 
    # the resampling, always between 50% and 90% of a uniform distribution)
    eps_min, eps_max = EPSILON_BOUNDS
    epsilon = epsilon or EPSILON
    eps_prec = 1e-3 
    niter = 0
    while True:
        niter += 1
        logging.debug('niter: {}, epsilon: {}'.format(niter, epsilon))
        if niter > 100: 
            logging.warning("too many iterations when estimating exponent")
            break
            # raise RuntimeError("too many iterations when estimating exponent")

        ratio_eps = _get_Neff(likelihood**epsilon) / N

        if epsilon < eps_min:
            logging.info('epsilon = {} < {} = eps_min. Set back to eps_min. Effective ensemble size too low : Neff/N = {}'.format(epsilon,eps_min,ratio_eps))
            epsilon = eps_min
            break
        if epsilon > eps_max:
            logging.info('epsilon = {} > {} = eps_max. Set back to eps_max. Effective ensemble size too high : Neff/N = {}'.format(epsilon,eps_max,ratio_eps))
            epsilon = eps_max
            break

        # neff_bounds = [0.5, 0.9]
        if ratio_eps > neff_bounds[1]:
            # Effective ensemble size too high, increase epsilon
            eps_incr = max(eps_prec, (eps_max - epsilon)/2)
            epsilon += eps_incr
        elif ratio_eps < neff_bounds[0]:
            # Effective ensemble size too low, decrease epsilon
            eps_incr = max(eps_prec, (epsilon - eps_min)/2)
            epsilon -= eps_incr
        else:
            break

    logging.info("...epsilon={} : Neff/N = {}".format(epsilon, ratio_eps))

    return epsilon


# Resampling
# ==========

def _build_ids(counts):
    """ make an array of ids from counts, e.g. [3, 0, 1] will returns [0, 0, 0, 2]
    """
    ids = np.empty(counts.sum(), dtype=int)
    start = 0
    for i, count in enumerate(counts):
        ids[start:start+count] = i
        start += count
    return ids

def multinomial_resampling(weights, size):
    """
    weights : (normalized) weights 
    size : sample size to draw from the weights
    """
    counts = np.random.multinomial(size, weights)
    return _build_ids(counts)

def residual_resampling(weights, size):
    """
    Deterministic resampling of the particles for the integer part of the counts
    Random sampling of the residual.
    Each particle (index) is copied int(weights[i]*size) times
    """
    # copy particles
    counts_decimal = weights * size
    counts_copy = np.asarray(np.floor(counts_decimal), dtype=int)
    # sample randomly from residual weights
    weights_resid = counts_decimal - counts_copy
    weights_resid /= weights_resid.sum()
    counts_resid = np.random.multinomial(size - counts_copy.sum(), weights_resid)
    # make the ids
    return _build_ids(counts_copy + counts_resid)

# Jitter step
# ===========

def sample_with_bounds_check(params, covjitter, bounds):
    """ Sample from covariance matrix and update parameters

    Parameters
    ----------
    params : 1-D numpy array (p)
    covjitter : covariance matrix p * p
    bounds : p*2 array (p x 2)
        parameter bounds: array([(min, max), (min, max), ...])

    Returns
    -------
    newparams : 1-D numpy array of resampled parameters
    """
    assert params.ndim == 1

    # prepare the jitter
    tries = 0
    maxtries = 100
    while True:
        tries += 1
        if seed is not None:
            np.random.seed(seed+tries) 
        newparams = np.random.multivariate_normal(params, covjitter)
        params_within_bounds = not np.any((newparams < bounds[:,0]) | (newparams > bounds[:,1]), axis=0)
        if params_within_bounds:
            logging.debug("Required {} time(s) sampling jitter to match bounds".format(tries, i))
            break
        if tries > maxtries : 
            logging.warning("Could not add jitter within parameter bounds")
            newparams = params
            break
    return newparams


def add_jitter(params, epsilon, bounds=None, seed=None):
    """ Add noise with variance equal to epsilon times ensemble variance

    params : size x p
    epsilon : float
    bounds : p x 2, optional
    """
    size = params.shape[0]
    covjitter = np.cov(params.T)*epsilon
    if covjitter.ndim == 0: 
        covjitter = covjitter.reshape([1,1]) # make it 2-D

    if seed is not None:
        np.random.seed(seed)  # NOTE: only apply to first sampling (bounds=None)
    jitter = np.random.multivariate_normal(np.zeros(params.shape[1]), covjitter, size)
    newparams = params + jitter

    # Check that params remain within physically-motivated "hard" bounds:
    if bounds is not None:
        bad = np.any((newparams < bounds[:,0][np.newaxis, :]) | (newparams > bounds[:,1][np.newaxis, :]), axis=0)
        ibad = np.where(bad)[0]
        if ibad.size > 0:
            logging.warning("{} particles are out-of-bound after jittering: resample within bounds".format(len(ibad)))
            # newparams[ibad] = resampled_params[ibad]
            for i in ibad:
                newparams[i] = sample_with_bounds_check(params[i], covjitter, bounds, seed=seed)

    return newparams


class Resampler(object):
    """Resampler class : wrap it all
    """
    def __init__(self, weights, normalize=True):
        if normalize:
            weights = weights / weights.sum()
        self.weights = weights

    def sample_residual(self, size):
        return residual_resampling(self.weights, size)

    def sample_multinomal(self, size):
        return multinomial_resampling(self.weights, size)

    def sample(self, size, seed=None, method=RESAMPLING_METHOD):
        """wrapper resampler method 
        """
        np.random.seed(seed) # random state
        if method == 'residual':
            ids = self.sample_residual(size)
        elif method == 'residual':
            ids = self.sample_multinomal(size)
        elif method in ("stratified", "deterministic"):
            raise NotImplementedError(method) # todo
        else:
            raise NotImplementedError(method)
        return np.sort(ids)  # sort indices (has no effect on the results)

    def neff(self):
        " effective ensemble size "
        return _get_Neff(self.weights, normalize=False)

    def size(self):
        return len(self.weights)

    def scaled(self, epsilon):
        """New resampler with scaled weights
        """
        return Resampler(self.weights**epsilon)

    def autoepsilon(self, neff_bounds=NEFF_BOUNDS, epsilon=EPSILON):
        """return epsilon to get effective ensemble size within bounds
        """
        return adaptive_posterior_exponent(self.weights, epsilon, neff_bounds)

    def iis(self, params, epsilon=None, size=None, bounds=None, seed=None, neff_bounds=NEFF_BOUNDS, **kwargs):
        """Iterative importance (re)sampling with scaled weights and jittering

        params : size x p
        epsilon : float
            weights <- weights ** epsilon
            see Resampler.autoepsilon
        bounds : p x 2, optional
            parameter bounds, force resampling if outside
        """
        if epsilon is None:
            epsilon = self.autoepsilon(neff_bounds)
        size = size or len(params)
        ids = self.scaled(epsilon).sample(size, seed=seed, **kwargs)
        return add_jitter(params[ids], epsilon, seed=seed, bounds=bounds)



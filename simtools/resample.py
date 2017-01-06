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
import argparse
from argparse import RawDescriptionHelpFormatter
import numpy as np


SAMPLING_METHOD = "residual"
EPSILON = 0.05  # start value for adaptive_posterior_exponent (kept if NEFF in NEFF_BOUNDS)
EPSILON_BOUNDS = (1e-3, 0.1)  # has priority over NEFF_BOUNDS
NEFF_BOUNDS = (0.5, 0.9)
DEFAULT_SIZE = 500
DEFAULT_ALPHA_TARGET = 0.95


def str_pmatrix(pnames, pmatrix, max_rows=10, include_index=True, index=None):
    """Pretty-print parameters matrix like in pandas, but using only basic python functions
    """
    # determine columns width
    col_width_default = 6
    col_fmt = []
    col_width = []
    for p in pnames:
        w = max(col_width_default, len(p))
        col_width.append( w )
        col_fmt.append( "{:>"+str(w)+"}" )

    # also add index !
    if include_index:
        idx_w = len(str(len(pmatrix)-1)) # width of last line index
        idx_fmt = "{:<"+str(idx_w)+"}" # aligned left
        col_fmt.insert(0, idx_fmt)
        pnames = [""]+list(pnames)
        col_width = [idx_w] + col_width

    line_fmt = " ".join(col_fmt)

    header = line_fmt.format(*pnames)

    # format all lines
    lines = []
    for i, pset in enumerate(pmatrix):
        if include_index:
            ix = i if index is None else index[i]
            pset = [ix] + list(pset)
        lines.append(line_fmt.format(*pset))

    n = len(lines)
    # full print
    if n <= max_rows:
        return "\n".join([header]+lines)

    # partial print
    else:
        sep = line_fmt.format(*['.'*min(3,w) for w in col_width])  # separator '...'
        return "\n".join([header]+lines[:max_rows//2]+[sep]+lines[-max_rows//2:])

def _get_Neff(weights):
    """ Return an estimate of the effective ensemble size
    """
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
    counts_copy = np.floor(counts_decimal)
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
    bounds : 2*p array (2 x p)
        parameter bounds: array([min1,min2,min3,...],[max1,max2,max3,...])

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
        newparams = np.random.multivariate_normal(params, covjitter)
        params_within_bounds = not np.any((newparams < bounds[0]) | (newparams > bounds[1]), axis=0)
        if params_within_bounds:
            logging.debug("Required {} time(s) sampling jitter to match bounds".format(tries, i))
            break
        if tries > maxtries : 
            logging.warning("Could not add jitter within parameter bounds")
            newparams = params
            break
    return newparams


def add_jitter(params, epsilon, bounds=None):
    """ Add noise with variance equal to epsilon times ensemble variance

    params : size x p
    epsilon : float
    bounds : 2 x p, optinal
    """
    size = params.shape[0]
    covjitter = np.cov(params.T)*epsilon
    if covjitter.ndim == 0: 
        covjitter = covjitter.reshape([1,1]) # make it 2-D
    jitter = np.random.multivariate_normal(np.zeros(params.shape[1]), covjitter, size)
    newparams = params + jitter

    # Check that params remain within physically-motivated "hard" bounds:
    if bounds is not None:
        bad = np.any((newparams < bounds[0][np.newaxis, :]) | (newparams > bounds[1][np.newaxis, :]), axis=0)
        ibad = np.where(bad)[0]
        if ibad.size > 0:
            logging.warning("{} particles are out-of-bound after jittering: resample within bounds".format(len(ibad)))
            # newparams[ibad] = resampled_params[ibad]
            for i in ibad:
                newparams[i] = sample_with_bounds_check(params[i], covjitter, bounds)

    return newparams


def bounds_parser(params_bounds):
    mi, ma = params_bounds[i].split(',')
    return float(mi), float(ma)

def read_params(pfile):
    pnames = open(pfile).readline().split()
    pvalues = np.loadtxt(pfile, skiprows=1)  
    return pnames, pvalues

def main():
    parser = argparse.ArgumentParser(description=__doc__,
	    formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('-w','--weights-file', required=True)
    parser.add_argument('--log', action='store_true', help='weights are provided as log-likelihood?')

    parser.add_argument('--resampling', choices=['residual', 'multinomial'], default=SAMPLING_METHOD, help='resampling method (default: %(default)s)')
    parser.add_argument('--include-index', action='store_true', help='include index in the output matrix')
    parser.add_argument('-N', '--size', help="New sample size (default: same size as before)", type=int)
    parser.add_argument('--seed', type=int, help="random seed, for reproducible results (default to None)")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('-p', '--params-file', help='parameter file of the ensemble')
    grp.add_argument('--index-only', action='store_true', help='only output the resampled index')
    grp.add_argument('--neff-only', action='store_true', help='only output effective ensemble size, given epsilon')
    grp.add_argument('--auto-epsilon-only', action='store_true', help='only output auto-epsilon')
    
    group = parser.add_argument_group('weights scaling')
    grp = group.add_mutually_exclusive_group()
    grp.add_argument('--epsilon', type=float, help='Exponent to flatten the weights.')
    grp.add_argument('--auto-epsilon', action='store_true', help='automatically determine epsilon')
    group.add_argument('--neff-bounds', nargs=2, default=NEFF_BOUNDS, type=int, help='effective ensemble size, to determine epsilon in "auto" mode')
    #parser.add_argument('--jitter', action='store_true', help='if True, add jitter to the ensemble after resampling, as epsilon times covariance')

    group = parser.add_argument_group('add noise')
    group.add_argument('--jitter', action='store_true', help='Add noise to the ensemble after resampling, with a fraction of the covariance (see --jitter-eps)')
    group.add_argument('--jitter-eps', type=float, help='if --jitter, fraction of original (flattened) covariance matrix. Default to epslion.')
    group.add_argument('--params-bounds', nargs='*', type=bounds_parser, help='list of min,max <= len(params)')
    parser.add_argument('--iis', action='store_true', help='Shortcut for --auto-epsilon and --jitter')

    #parser.add_argument('-o', '--out', help="Output parameter file")

    args = parser.parse_args()

    weights = np.loadtxt(args.weights_file)

    if args.log:
        weights = np.exp(weights)

    if args.iis:
        if not args.epsilon:
            args.auto_epsilon = True
        args.jitter = True

    if args.auto_epsilon_only:
        args.auto_epsilon = True

    if args.auto_epsilon:
        args.epsilon = adaptive_posterior_exponent(weights, neff_bounds=args.neff_bounds)

    if args.auto_epsilon_only:
        print(args.epsilon)
        return

    if args.epsilon:
        weights = weights**args.epsilon

    if args.neff_only:
        print( _get_Neff(weights) )
        return

    size = args.size or weights.size

    weights /= weights.sum() # normalize weights

    # Resample the model versions according to their weight
    # -----------------------------------------------------
    np.random.seed(args.seed)
    if args.resampling == "multinomial":
        ids = multinomial_resampling(weights, size)
    elif args.resampling == "residual":
        ids = residual_resampling(weights, size)
    elif args.resampling in ("stratified", "deterministic"):
        raise NotImplementedError(args.resampling)
    else:
        raise ValueError("Unknown resampling method: "+args.resampling)
    ids = np.sort(ids)  # sort indices (has no effect on the results)


    if args.index_only:
        print(ids)
        return


    # Resample parameters
    # -------------------
    pnames, params = read_params(args.params_file)
    newparams = params[ids]

    # add jitter
    if args.jitter:
        eps = args.jitter_eps or args.epsilon
        assert eps, 'if need provide jitter-eps or epsilon'

        if args.params_bounds:
            missing = params.shape[1]*len(args.params_bounds)  # only first values provided?
            args.params_bounds = args.params_bounds + [(-np.inf, np.inf)]*missing

        np.random.seed(args.seed)  # NOTE: only apply to first sampling (bounds=None)
        newparams = add_jitter(newparams, eps, bounds=args.params_bounds)

    txt = str_pmatrix(pnames, newparams, index=ids, include_index=args.include_index, max_rows=np.inf)
    print(txt)

if __name__ == '__main__':
    main()


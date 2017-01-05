#!/usr/bin/env python2.7
"""Compute cost-function for a model output
"""
from __future__ import print_function, division
import os, sys, datetime, warnings

from itertools import product
from collections import OrderedDict as odict

import numpy as np
import scipy.stats
from scipy.stats import norm, lognorm, uniform, multivariate_normal


def parse_scipydist(string):
    """parse a scipy distribution:  NAME?P1,P2,...

    e.g. norm?0,1 or uniform?-10,10 or lognorm?1,0,1
    """
    dtype, spec = string.split('?')
    params = [float(p) for p in spec.split(',')]
    return getattr(scipy.stats, dtype)(*params)


class Constraint(object):
    """Generic constraint class, assumes variable in a netCDF file.
    """
    def __init__(self, name, dist, units="", desc=""):


        self.name = name
        self.units = units
        self.desc = desc
        self.dist = dist

    def logpdf(self, state):
        return self.dist.logpdf(state)

    def mean(self):
        return self.dist.mean()


class Uniform(Constraint):
    def __init__(self, name, low, hi, units="", desc=""):
        Constraint.__init__(self, name, uniform(low, hi), units, desc) 


class Normal(Constraint):
    def __init__(self, name, mean, sd, units="", desc=""):
        Constraint.__init__(self, name, norm(mean, sd), units, desc) 


class LogNormal(Constraint):
    """scipy.stats.lognorm """
    def __init__(self, name, s, loc, scale, units="", desc=""):
        Constraint.__init__(self, name, lognorm(s, loc, scale), units, desc) 


class RMS(Constraint):
    """Root Mean Square : treat as a single observation
    """
    def __init__(self, name, mean, sd, mask=None, units="", desc=""):
        self.name = name 
        self.units = units
        self.desc = desc
        self.mean = mean
        self.var = sd**2
        self.mask = mask

    def logpdf(self, state):
        # overwrite standard method to handle difference in grid
        assert state.size == self.mean.size, "array size do not match"
        if self.mask is None:
            mask = np.ones(n, dtype=bool)
        else:
            mask = self.mask
        misfit = state[mask] - self.mean[mask]
        var = self.var[mask]
        return -0.5 * np.mean( misfit ** 2 / var ) 


class MultiVariateNormal(Constraint):
    """Account for covariances
    """
    def __init__(self, name, mean, cov, units="", desc=""):
        dist = multivariate_normal(mean, cov=cov)
        Constraint.__init__(self, name, dist, units, desc) 


def parse_constraint(*args):
    " name=... type=... units=... desc=... mean=... sd=... mean_var=... sd_var... file=... valid=... etc..."

    kwargs = {}
    for i, arg in enumerate(args):
        nm, val = arg.split('=')
        kwargs[nm] = val

    name = kwargs.pop('name')
    dtype = kwargs.pop('type')

    if '?' in dtype:
        dtype, spec = dtype.split('?')
    else:
        spec = ''
    
    if dtype == 'norm':
        loc, scale = spec.split(',')
        constraint = Normal(name, float(loc), float(scale), **kwargs)

    elif dtype == 'uniform':
        loc, scale = spec.split(',')
        constraint = Uniform(name, float(loc), float(scale), **kwargs)

    elif dtype == 'lognorm':
        s, loc, scale = spec.split(',')
        constraint = LogNormal(name, float(s), float(loc), float(scale), **kwargs)

    elif dtype == 'rms':
        assert not spec, 'invalid specificiation for rms type, use keywords mean, sd, mean_var, sd_var'
        try:
            constraint = RMS.parse(name, dtype, **kwargs)
        except:
            print(RMS.parse.__doc__)
            raise

    else:
        raise ValueError("Unknown constraint type: "+dtype)

    return constraint


def get_ensemble_size(outdir):
    import glob
    direcs = sorted(glob.glob(os.path.join(outdir,'0*/')))
    lastid = os.path.basename(direcs[-1].rstrip('/'))
    return int(lastid)+1  # assum start at 0 (in case some intermediary direcs are missing)


def ensemble_loglik(states, constraints):
    """read output state variables

    constraints: model output
    runids: sub-sample of indices to read (default: all)
    N: size of the ensemble
    
    OUTDIR/ID/restart.nc
    """
    N = len(states)
    m = len(constraints)

    loglik = np.empty((N, m))
    loglik.fill(-np.inf)
    
    for i, state in enumerate(states):
        for j, c in enumerate(constraints):
            if state[j] is not None:
                loglik[i,j] = c.logpdf(state[j])

    return loglik

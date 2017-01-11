#!/usr/bin/env python2.7
"""Compute cost-function for a model output
"""
from __future__ import print_function, division
import os, warnings
import json

import numpy as np
import scipy.stats
from scipy.stats import norm, lognorm, uniform, multivariate_normal


def parse_scipy(spec):
    """parse scipy distribution, e.g. norm?loc,scale or lognorm?shp,loc,scale
    """
    type, args = spec.split("?")
    params = [float(p) for p in args.split(',')]
    return getattr(scipy.stats, type)(*params)


class Constraint(object):
    """Constraint as normal distribution
    """
    def __init__(self, name, mean, std, units="", desc=""):
        self.name = name
        self.mean = mean
        self.std = std
        self.var = std**2
        self.units = units
        self.desc = desc

    def logpdf(self, state):
        return -0.5*(state - self.mean)/self.var

    def __call__(self, state):
        """Actual weight
        """
        return np.exp(self.logpdf(state))


class Normal(Constraint):
    pass


class ScipyConstraint(Constraint):
    """Generic constraint class, assumes variable in a netCDF file.
    """
    def __init__(self, name, dist, units="", desc=""):
        self.name = name
        self.mean = dist.mean()
        self.std = dist.std() if hasattr(dist, 'std') else None
        self.units = units
        self.desc = desc
        self.dist = dist

    def logpdf(self, state):
        return self.dist.logpdf(state)


class MultiVariateNormal(ScipyConstraint):
    """Account for covariances
    """
    def __init__(self, name, mean, cov, **kwargs):
        dist = multivariate_normal(mean, cov=cov)
        ScipyConstraint.__init__(self, name, dist, **kwargs)


class Uniform(ScipyConstraint):
    def __init__(self, name, lo, hi, units="", desc=""):
        ScipyConstraint.__init__(self, name, uniform(lo, hi-lo), units, desc) 


class LogNormal(ScipyConstraint):
    """scipy.stats.lognorm """
    def __init__(self, name, s, loc, scale, units="", desc=""):
        ScipyConstraint.__init__(self, name, lognorm(s, loc, scale), units, desc) 


class RMS(Constraint):
    """Root Mean Square : treat as a single observation
    """
    def __init__(self, name, mean, sd, mask=None, **kwargs):
        Constraint.__init__(self, name, mean, sd, **kwargs)
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


def parse_constraint(string, getobs=None):
    """Parse constraint

    string: name=spec 
        where name is the name of the constraint (which can be passed to getobs)
        and spec can be one of:
        [scipy dist] name=type?loc,scale,... 
        [custom] name=error
        where `error` indicates the standard error for a normal distribution.
        `error` is either a number or can be defined as a percentage of the 
        mean obs value (just append a `%` sign). If getobs returns an array, 
        the (minus) RMS will be returned for loglikehood. 
        Note that `error` basically affects how much each observation is weighted
        compared to the other.

    getobs: callable, opt
        if "%" is present, or if RMS is provided, will be used
        to call the concurrent observation getobs(name)
    """
    name, spec = string.split("=")

    # Parse constraint as scipy distribution, e.g. norm?loc,scale
    if '?' in spec:
        dist = parse_scipy(spec)
        return ScipyConstraint(name, dist)

    # custom
    obs = getobs(name)
    if spec.endswith('%'):
        err = float(spec[:-1])*obs/100
    else:
        err = float(spec)

    if np.size(obs) > 1:
        c = RMS(name, obs, err)
    else:
        c = Normal(name, obs, err)

    return c


def get_constraint(name, error=None, pct_error=None, getobs=None, **kwargs):
    """Return a list of Constraints based on config
    """
    obs = getobs(name)
    if pct_error:
        error = obs*pct_error/100

    if np.size(obs) > 1:
        c = RMS(name, obs, error, **kwargs)
    else:
        c = Normal(name, obs, error, **kwargs)

    return c




class Likelihood(Constraint):
    def __init__(self, constraints):
        """Likelihood as composite of several constraints
        """
        self.constraints = constraints

    def names(self):
        return [c.name for c in self.constraints]

    @property
    def mean(self):
        return [c.mean for c in self.constraints]

    @property
    def std(self):
        return [c.std for c in self.constraints]

    def __getitem__(self, name):
        return self.constraints[self.names().index(name)]

    @classmethod
    def read(cls, file, getobs=None):
        dat = json.load(open(file))
        constraints = [get_constraint(getobs=getobs, **cdef) 
                       for cdef in dat["constraints"]]
        return cls(constraints)

    def update(self, constraints):
        for c in constraints:
            names = self.names()
            if c.name in names:
                self.constraints[names.index[c.name]] = c  # replace
            else:
                self.constraints.append(c)

    def logpdf(self, state):
        return sum([c.logpdf(s) for c, s in zip(self.constraints, state)])


#def get_ensemble_size(outdir):
#    import glob
#    direcs = sorted(glob.glob(os.path.join(outdir,'0*/')))
#    lastid = os.path.basename(direcs[-1].rstrip('/'))
#    return int(lastid)+1  # assum start at 0 (in case some intermediary direcs are missing)
#

#def ensemble_loglik(states, constraints):
#    """read output state variables
#
#    constraints: model output
#    runids: sub-sample of indices to read (default: all)
#    N: size of the ensemble
#    
#    OUTDIR/ID/restart.nc
#    """
#    N = len(states)
#    m = len(constraints)
#
#    loglik = np.empty((N, m))
#    loglik.fill(-np.inf)
#    
#    for i, state in enumerate(states):
#        for j, c in enumerate(constraints):
#            if state[j] is not None:
#                loglik[i,j] = c.logpdf(state[j])
#
#    return loglik

#!/usr/bin/env python2.7
"""Compute cost-function for a model output
"""
from __future__ import print_function, division
import os, warnings
import json

import numpy as np
import scipy.stats
from scipy.stats import norm, lognorm, uniform, multivariate_normal
from simtools.xrun import XRun, XDir
from simtools.tools import parse_dist, dist_to_str
from simtools.prior import PriorParam


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




#class Likelihood(Constraint):
#    def __init__(self, constraints):
#        """Likelihood as composite of several constraints
#        """
#        self.constraints = constraints
#
#    def names(self):
#        return [c.name for c in self.constraints]
#
#    @property
#    def mean(self):
#        return [c.mean for c in self.constraints]
#
#    @property
#    def std(self):
#        return [c.std for c in self.constraints]
#
#    def __getitem__(self, name):
#        return self.constraints[self.names().index(name)]
#
#    @classmethod
#    def read(cls, file, getobs=None):
#        dat = json.load(open(file))
#        constraints = [get_constraint(getobs=getobs, **cdef) 
#                       for cdef in dat["constraints"]]
#        return cls(constraints)
#
#    def update(self, constraints):
#        for c in constraints:
#            names = self.names()
#            if c.name in names:
#                self.constraints[names.index[c.name]] = c  # replace
#            else:
#                self.constraints.append(c)
#
#    def logpdf(self, state):
#        return sum([c.logpdf(s) for c, s in zip(self.constraints, state)])


class Analyzer(XDir):
    """perform analysis of the ensemble
    """
    def __init__(self, constraints, expdir):

    # state variables I/O
    def write_state_var(self, name, value, runid=None):
        """Write state variable on disk in a format understood by XRun
        """
        statefile = self.statefile(runid)+'.json' # state file in json format
        with open(statefile, "w") as f:
            json.dump(value, f)

    def read_state_var(self, name, runid=None):
        statefile = self.statefile(runid)+'.json' # state file in json format
        with open(statefile) as f:
            return json.load(f)


    # analyze ensemble
    # ----------------
    def get(self, name, runid=None):
        """Get variable 
        """
        return self.read_state_var(name, runid)


    def get_all(self, name):
        """Return variable for all realizations
        """
        dim = size(self.get(name, 0)) # check size of first variable
        var = np.empty((self.params.size, dim))
        var.fill(np.nan)
        for i in xrange(self.params.size):
            var[i] = self.get(name, i)
        return var.squeeze(1)


    def loglik(self, constraints, runid=None):
        """Log-like for one realization
        """
        return sum([c.logpdf( self.get(c.name, runid)) for c in constraints])


    def loglik_all(self, constraints):
        """Log-likelihood for all realizations
        """
        var = np.empty(self.params.size)
        for i in xrange(self.params.size):
            try:
                var[i] = self.loglik(constraints, i)
            except:
                var[i] = -np.inf
        return var

    
    def analyze(self, constraints, fill_array=np.nan):
        """Analyze experiment directory and return a Results objet

        Parameters
        ----------
        constraints : list of constraints
        fill_array : float or callable
            value to use instead of (skipped) array constraints (nan by default)
        """
        from simtools.analysis import Results

        N = self.params.size
        state2 = np.empty((N, len(constraints)))
        state2.fill(np.nan)
        loglik2 = np.empty((N, len(constraints)))
        loglik2.fill(-np.inf)

        def reduce_array(s):
            return fill_array(s) if callable(fill_array) else fill_array

        failed = 0

        for i in xrange(N):
            try:
                state = [self.get(c.name, i) for c in constraints]
            except Exception as error:
                failed += 1
                continue

            # diagnostic per constraint
            for j, s in enumerate(state):
                loglik2[i, j] = constraints[j].logpdf(s)
                state2[i, j] = s if np.size(s) == 1 else reduce_array(s)

        print("warning :: {} out of {} simulations failed".format(failed, N))

        return Results(constraints, state2, loglik2=loglik2, params=self.params)


class Results(object):
    """Contains model result for further analysis with constraints
    """
    def __init__(self, constraints=None, state=None, 
                 loglik=None, loglik2=None, params=None, default=None):

        if loglik is None and loglik2 is not None:
            loglik = loglik2.sum(axis=1)
        self.loglik = loglik
        self.constraints = constraints   # so that names is defined

        self.state = state
        self.loglik2 = loglik2
        self.params = params
        self.default = default

        # weights
        self.loglik = loglik
        self.valid = np.isfinite(self.loglik)

    def weights(self):
        w = np.exp(self.loglik)
        return w / w.sum()

    @classmethod
    def read(cls, direc):
        x = XDir(direc)
        loglik = np.loadtxt(x.path("loglik.txt"))
        return cls(loglik=loglik)


    def write(self, direc):
        """write result stats and loglik to folder
        """
        print("write analysis results to",direc)
        x = XDir(direc)
        np.savetxt(x.path("loglik.txt"), self.loglik)

        if self.state is not None:
            with open(x.path("state.txt"), "w") as f:
                f.write(self.format(self.state))
            with open(x.path("stats.txt"), "w") as f:
                f.write(self.stats())

        if self.loglik2 is not None:
            with open(x.path("loglik.all.txt"), "w") as f:
                f.write(self.format(self.loglik2))


    @property
    def obs(self):
        return [c.mean for c in self.constraints]

    @property
    def names(self):
        return [c.name for c in self.constraints]

    def best(self):
        return self.state[np.argmax(self.loglik)]

    def mean(self):
        return self.state[self.valid].mean(axis=0)

    def std(self):
        return self.state[self.valid].std(axis=0)

    def min(self):
        return self.state[self.valid].min(axis=0)

    def max(self):
        return self.state[self.valid].max(axis=0)

    def pct(self, p):
        return np.percentile(self.state[self.valid], p, axis=0)


    def stats(self, fmt="{:.2f}", sep=" "):
        """return statistics
        """
        #def stra(a):
        #    return sep.join([fmt.format(k) for k in a]) if a is not None else "--"

        res = [
            ("obs", self.obs),
            ("best", self.best()),
            ("default", self.default),
            ("mean", self.mean()),
            ("std", self.std()),
            ("min", self.min()),
            ("p05", self.pct(5)),
            ("med", self.pct(50)),
            ("p95", self.pct(95)),
            ("max", self.max()),
        ]

        index = [nm for nm,arr in res if arr is not None]
        values = [arr for nm,arr in res if arr is not None]

        import pandas as pd
        df = pd.DataFrame(np.array(values), columns=self.names, index=index)

        return str(df) #"\n".join(lines)

    def df(self, array):
        " transform array to dataframe "
        import pandas as pd
        return pd.DataFrame(array, columns=self.names)

    def format(self, array):
        return str(self.df(array))




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

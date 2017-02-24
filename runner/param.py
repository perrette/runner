"""Parameter or state variable as random variable
"""
from __future__ import division
import json
import logging
import sys
import itertools
from collections import OrderedDict as odict
import numpy as np

import runner.xparams as xp
from runner.xparams import XParams
from runner.lib.doelhs import lhs
from runner.tools.dist import parse_val, DiscreteDist, cost
from runner.tools.dist import parse_dist2, dist_to_str2, dist_todict2, dist_fromkw2

# default criterion for the lhs method
LHS_CRITERION = 'centermaximin' 

# for reading...
ALPHA = 0.99  # validity interval

class Param(object):
    """random variable: parameter or state var
    """
    def __init__(self, name, default=None, dist=None, help=None, full_name=None):
        """
        * name 
        * dist : scipy distribution - like
        * help : parameter info
        * full_name : to be used for file I/O (e.g. namelist, includes prefix)
        """
        self.name = name
        self.dist = dist
        self.default = default
        self.help = help
        self.full_name = full_name

    def __call__(self, value=None):
        return FrozenParam(self, value)

    def __str__(self):
        #return "{name}={value}".format(name=self.name, value=self.value)
        if self.dist:
            return "{name}={dist}".format(name=self.name, dist=dist_to_str2(self.dist))
        else:
            return "{name}={default}".format(name=self.name, default=self.default)

    def __eq__(self, other):
        return (isinstance(other, Param) and self.name == other.name) \
            or (isinstance(other, six.string_types) and self.name == other)


    @classmethod
    def parse(cls, string):
        """Prior parameter defintion as NAME=SPEC.

        SPEC specifies param values or distribution.
        Discrete parameter values can be provided 
        as a comma-separated list `VALUE[,VALUE...]`
        or a range `START:STOP:N`.
        A distribution is provided as `TYPE?ARG,ARG[,ARG,...]`.
        Pre-defined `U?min,max` (uniform) and `N?mean,sd` (normal)
        or any scipy.stats distribution as TYPE?[SHP,]LOC,SCALE.
        """
        # otherwise custom, command-line specific representation
        try:
            name, spec = string.split('=')
            if '!' in spec:
                spec, default = spec.split('!')
                default = parse_val(default)
            else:
                default = None
            dist = parse_dist2(spec)
            return cls(name, dist=dist, default=default)

        except Exception as error:
            logging.error(str(error))
            raise


    def as_dict(self):
        kw = self.__dict__.copy()
        dist = kw.pop('dist')
        kw2 = dist_todict2(dist)
        for k in kw2:
            kw['dist_'+k] = kw2[k]
        return {k:v for k,v in kw.items() if v is not None}

    @classmethod
    def fromkw(cls, name, **kwargs):
        kw2 = {}
        for k in kwargs.keys():
            if k.startswith('dist_'):
                kw2[k[5:]] = kwargs.pop(k)
        if kw2:
            dist = dist_fromkw2(**kw2)
        else:
            dist = None
        return cls(name, dist=dist, **kwargs)


class FrozenParam(object):
    """Parameter / State variable with fixed value
    """
    def __init__(self, param, value=None):
        self.param = param
        self.value = value if value is not None else param.default

    @property
    def name(self):
        return self.param.name

    @property
    def dist(self):
        " scipy or custom distribution (frozen) "
        return self.param.dist if self.param.dist else dummydist(self.default)

    def __str__(self):
        if self.value is None:
            val = '({})'.format(self.param.default)
        else:
            val = self.value
        return "{}={} ~ {}".format(self.name, val, self.dist)

    # distribution applied to self:
    def logpdf(self):
        return self.dist.logpdf(self.value)

    def pdf(self):
        return self.dist.pdf(self.value)

    def isvalid(self, alpha=ALPHA):
        """params in the confidence interval
        """
        lo, hi = self.dist.interval(alpha)
        if not np.isfinite(self.value) or self.value < lo or self.value > hi:
            return False
        else:
            return True

    # back-compat
    # TODO: remove
    @property
    def cost(self):
        return cost(self.dist, self.value) if np.isfinite(self.value) else np.inf


# parsing made easier
class DiscreteParam(Param):
    def __init__(self, *args, **kwargs):
        super(DiscreteParam, self).__init__(*args, **kwargs)
        if not isinstance(self.dist, DiscreteDist):
            raise TypeError("expected DiscreteDist, got: "+type(self.dist).__name__)


class ScipyParam(Param):
    def __init__(self, *args, **kwargs):
        super(ScipyParam, self).__init__(*args, **kwargs)
        if isinstance(self.dist, DiscreteDist):
            raise TypeError("expected scipy dist, got discrete values")
        

def filterkeys(kwargs, keys):
    return {k:kwargs[k] for k in kwargs if k in keys}


class ParamList(list):
    """enhanced list: pure python data structure, does not do any work
    """
    def __init__(self, params):
        " list of Param instances"
        super(ParamList, self).__init__(params)
        for p in self:
            if not hasattr(p, 'name'):
                raise TypeError("Param-like with 'name' attribute required, got:"+repr(type(p)))

    @property
    def names(self):
        return [p.name for p in self]

    def __getitem__(self, name):
        if type(name) is int:
            return super(ParamList, self)[name]
        else:
            return {p.name:p for p in self}[name]


    def __add__(self, other):
        return type(self)(list(self) + list(other))


class MultiParam(ParamList):
    """Combine a list of parameters or state variables, can sample, compute likelihood etc
    """

    def product(self):
        for p in self:
            if not isinstance(p.dist, DiscreteDist):
                raise TypeError("cannot make product of continuous distributions: "+p.name)
        return XParams(list(itertools.product(*[p.dist.values.tolist() for p in self])), self.names)


    def sample_montecarlo(self, size, seed=None):
        """Basic montecarlo sampling --> return pmatrx
        """
        pmatrix = np.empty((size,len(self.names)))

        for i, p in enumerate(self):
            pmatrix[:,i] = p.dist.rvs(size=size, random_state=seed+i if seed else None) # scipy distribution: sample !

        return XParams(pmatrix, self.names)


    def sample_lhs(self, size, seed=None, criterion=LHS_CRITERION, iterations=None):
        """Latin hypercube sampling --> return Xparams
        """
        pmatrix = np.empty((size,len(self.names)))
        np.random.seed(seed)
        lhd = lhs(len(self.names), size, criterion, iterations) # sample x parameters, all in [0, 1]

        for i, p in enumerate(self):
            pmatrix[:,i] = p.dist.ppf(lhd[:,i]) # take the quantile for the particular distribution

        return XParams(pmatrix, self.names)


    def sample(self, size, seed=None, method="lhs", **kwargs):
        """Wrapper for the various sampling methods. Unused **kwargs are ignored.
        """
        pmatrix = np.empty((size,len(self.names)))
        if method == "lhs":
            opts = filterkeys(kwargs, ['criterion', 'iterations'])
            xparams = self.sample_lhs(size, seed, **opts)
        else:
            xparams = self.sample_montecarlo(size, seed)
        return xparams

    def __call__(self, **kw):
        return FrozenParams([p(kw.pop(p.name, p.default)) for p in self])


    def asdict(self, key=None):
        return {key:[p.as_dict() for p in self]}

    @classmethod
    def fromdict(cls, kwds, key=None):
        return cls([Param.fromkw(p) for p in kwds[key]])



class FrozenParams(ParamList):

    def as_dict(self):
        return odict([(p.name,p.value) for p in self if p.value is not None])

    def logpdf(self):
        #if np.isfinite(self.getvalue()) else 0.
        return np.array([p.logpdf() for p in self])

    def pdf(self):
        return np.array([p.pdf for p in self])

    def isvalid(self, alpha=ALPHA):
        return np.array([p.isvalid(alpha) for p in self])

    # back-compat
    def cost(self):
        return np.array([p.cost for p in self])

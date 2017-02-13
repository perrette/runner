"""Parameter or state variable as random variable
"""
from __future__ import division
import json
import logging
import sys
import difflib
import itertools
import numpy as np

from runner.xparams import XParams
from runner.tools import parse_dist as parse_scipy, parse_list, parse_range, dist_to_str as scipy_to_str, LazyDist, dist_todict as scipy_todict, dist_fromkw as scipy_fromkw
from runner.lib.doelhs import lhs

import runner.xparams as xp

# default criterion for the lhs method
LHS_CRITERION = 'centermaximin' 

# for reading...
ALPHA = 0.99  # validity interval

# emulate scipy dist
class DiscreteDist(object):
    """Prior parameter that takes a number of discrete values
    """
    def __init__(self, values):
        self.values = np.asarray(values)

    def rvs(self, size):
        indices = np.random.randint(0, len(self.values), size)
        return self.values[indices]

    def ppf(self, q, interpolation='nearest'):
        return np.percentile(self.values, q*100, interpolation=interpolation)

    def __str__(self):
        return ",".join(*[str(v) for v in self.values])

    @classmethod
    def parse(cls, string):
        if ':' in string:
            values = parse_range(string)
        else:
            values = parse_list(string)
        return cls(values)


def parse_dist(string):
    if '?' in string:
        return parse_scipy(string)
    else:
        return DiscreteDist.parse(string)

def dist_to_str(dist):
    if isinstance(dist, DiscreteDist):
        return str(dist)
    else:
        return scipy_to_str(dist)


class Param(object):
    """random variable: parameter or state var
    """
    def __init__(self, name, value=None, dist=None, default=None, help=None, full_name=None, group=None):
        """
        * name 
        * value
        * dist : scipy distribution - like
        * default : default value
        * help : parameter info
        * full_name : to be used for file I/O (e.g. namelist, includes prefix)
        * group : could be used to specify correlations between parameters
        """
        self.name = name
        self.value = value
        self.dist = dist
        self.default = default
        self.help = help
        self.full_name = full_name
        self.group = group


    def __str__(self):
        #return "{name}={value}".format(name=self.name, value=self.value)
        if self.value:
            return "{name}={value}".format(name=self.name, value=self.value)
        elif self.dist:
            return "{name}={dist}".format(name=self.name, dist=dist_to_str(self.dist))
        else:
            return "{name}={default}".format(name=self.name, default=self.default)

    def __eq__(self, other):
        return isinstance(other, Param) and self.name == other.name

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
            dist = parse_dist(spec)
            return cls(name, dist=dist, default=default)

        except Exception as error:
            logging.error(str(error))
            raise


    # distribution applied to self:
    def logpdf(self):
        return self.dist.logpdf(self.value) if np.isfinite(self.value) else -np.inf

    def pdf(self):
        return self.dist.pdf(self.value) if np.isfinite(self.value) else 0.

    def isvalid(self, alpha=ALPHA):
        """params in the confidence interval
        """
        lo, hi = self.dist.interval(alpha)
        if not np.isfinite(self.value) or self.value < lo or self.value > hi:
            return False
        else:
            return True

    # back-compat
    def cost(self):
        return cost(self.dist, self.value) if np.isfinite(self.value) else np.inf

    def as_dict(self):
        kw = self.__dict__.copy()
        dist = kw.pop('dist')
        kw2 = dist_todict(dist)
        for k in kw2:
            kw['dist_'+k] = kw2[k]
        return kw

    @classmethod
    def fromkw(cls, name, **kwargs):
        kw2 = {}
        for k in kwargs:
            if k.startswith('dist_'):
                kw2[k[5:]] = kwargs.pop(k)
        if kw2:
            dist = dist_fromkw(**kw2)
        else:
            dist = None
        return cls(name, dist=dist, **kwargs)


def dist_todict(dist):
    if isinstance(dist, DiscreteDist):
        return {'values':dist.values.tolist(), 'name':'discrete'}
    return scipy_todict(dist)

def dist_fromkw(name, **kwargs):
    if name == 'discrete':
        return DiscreteDist(**kwargs)
    return scipy_fromkw(name, **kwargs)

def cost(dist, value):
    " logpdf = -0.5*cost + cte, only makes sense for normal distributions "
    logpdf = dist.logpdf(value)
    cst = dist.logpdf(dist.mean())
    return -2*(logpdf - cst)


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
        



# Commented out because the LHS topic is in fact non-trivial
# and involves correcting for space uniformity in the multi-
# dimensional space (e.g. see orthogonal lhs). The case below
# is a centered LHS where the only degree of randomness stems 
# from shuffling intervals. Fair enough but insatisfactory in 
# multiple dimensions.
#
#    def sample_lhs(self, size):
#        """Latin hypercube sampling distribution
#        """
#        qe = np.linspace(0, 1, size+1)
#        qc = (qe[:size] + qe[size:])/2
#        q = self.quantile(qc)
#        return np.random.shuffle(q)



def filterkeys(kwargs, keys):
    return {k:kwargs[k] for k in kwargs if k in keys}


class ParamList(list):
    """enhanced list: pure python data structure, does not do any work
    """
    def __init__(self, params):
        " list of Param instances"
        super(ParamList, self).__init__(params)
        for p in self:
            if not isinstance(p, Param):
                raise TypeError("expected Param, got:"+repr(type(p)))

    @property
    def names(self):
        return [p.name for p in self]

    def as_dict(self):
        return {p.name : p.value for p in self}


    def __getitem__(self, name):
        if type(name) is int:
            return super(ParamList, self)[name]
        else:
            return {p.name:p for p in self}[name]


    def __add__(self, other):
        return type(self)(list(self) + list(other))


    def update(self, params_kw, strict=False, verbose=True):
        """update with a dict (key, value) (inplace)
        """
        if not isinstance(params_kw, dict):
            raise TypeError('update params/state :: expected dict, got: {}'.format(type(params_kw).__name__))

        for name in params_kw:
            value = params_kw[name]

            if name in self.names:
                self[name].value = value

            elif not strict: 
                self.append(Param(name, value=value))

            else:
                if verbose:
                    logging.error("Available parameters:"+" ".join(self.names))
                    suggestions = difflib.get_close_matches(name, self.names)
                    if suggestions:
                        logging.error("Did you mean: "+ ", ".join(suggestions)+ " ?")
                raise


    def filter(self, predicate=None):
        """filter params, by default all that have a distribution defined
        """
        if predicate is None:
            predicate = lambda p : p.dist is not None

        elif isinstance(predicate, list):
            names = predicate
            predicate = lambda p : p.name in names

        params = [p for p in self if predicate(p)]
        return type(self)(params)



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


    def logpdf(self):
        return np.array([p.logpdf() for p in self])

    def pdf(self):
        return np.array([p.pdf() for p in self])

    def isvalid(self, alpha=ALPHA):
        return np.array([p.isvalid(alpha) for p in self])

    # back-compat
    def cost(self):
        return np.array([p.cost() for p in self])

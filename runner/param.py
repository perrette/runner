"""Parameter or state variable as random variable
"""
from __future__ import division
import json
from itertools import product
import logging
import sys
import numpy as np

import scipy.stats
from scipy.stats import norm, uniform, rv_continuous

from runner.tools import parse_dist as parse_scipy, parse_list, parse_range, dist_to_str as scipy_to_str
from runner.lib.doelhs import lhs

import runner.xparams as xp
from runner.xparams import XParams

# default criterion for the lhs method
LHS_CRITERION = 'centermaximin' 

# for reading...
PRIOR_KEY = "prior"


# emulate scipy dist
class DiscreteDist(object):
    """Prior parameter that takes a number of discrete values
    """
    def __init__(self, values):
        self.values = np.asarray(values)
        self.name = 'discrete'
    
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


    @classmethod
    def parse(cls, string):
        """Prior parameter defintion as NAME=SPEC.

        SPEC specifies param values or distribution.
        Discrete parameter values can be provided 
        as a comma-separated list `VALUE[,VALUE...]`
        or a range `START:STOP:N`.
        A distribution is provided as `TYPE?ARG,ARG[,ARG,...]`.
        Pre-defined `U?min,max` (uniform) and `N?mean,sd` (normal)
        or any scipy.stats distribution as TYPE?[SHP,]LOC,SCALE.")
        Additionally default value can be indicated with '!DEFAULT'
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


class DiscreteParam(Param):
    def __init__(self, *args, **kwargs):
        super(DiscreteParam, self).__init__(*args, **kwargs)
        if not isinstance(self.dist, DiscreteDist):
            raise TypeError("expected DiscreteDist, got: "+type(self.dist).__name__)



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



# json-compatible I/O
# ===================

def filterkeys(kwargs, keys):
    return {k:kwargs[k] for k in kwargs if k in keys}


class Params(object):
    def __init__(self, params):
        " list of Param instances (for product)"
        self.params = list(params)
        for p in self.params:
            if not isinstance(p, Param):
                raise TypeError("expected Param, got:"+repr(type(p)))

    #@classmethod
    #def read(cls, file, key=PRIOR_KEY, param_cls=Param):
    #    """read from config file

    #    file : json file
    #    key : sub-part of a larger json file?
    #    param_cls : optional, e.g. pick only Param or DiscreteDist
    #        (for more informative error messages)
    #    """
    #    cfg = json.load(open(file))
    #    if key and key in cfg: cfg = cfg[key]
    #    params = [param_cls.fromjson(json.dumps(p)) for p in cfg["params"]]
    #    return cls(params)


    @property
    def names(self):
        return [p.name for p in self.params]

    def sample_montecarlo(self, size):
        """Basic montecarlo sampling --> return XParams
        """
        pmatrix = np.empty((size,len(self.names)))

        for i, p in enumerate(self.params):
            pmatrix[:,i] = p.dist.rvs(size=size) # scipy distribution: sample !

        return XParams(pmatrix, self.names)

    def sample_lhs(self, size, criterion=LHS_CRITERION, iterations=None):
        """Latin hypercube sampling --> return Xparams
        """
        #from pyDOE import lhs

        pmatrix = np.empty((size,len(self.names)))
        lhd = lhs(len(self.names), size, criterion, iterations) # sample x parameters, all in [0, 1]

        for i, p in enumerate(self.params):
            pmatrix[:,i] = p.dist.ppf(lhd[:,i]) # take the quantile for the particular distribution

        return XParams(pmatrix, self.names)

    def sample(self, size, seed=None, method="lhs", **kwargs):
        """Wrapper for the various sampling methods. Unused **kwargs are ignored.
        """
        pmatrix = np.empty((size,len(self.names)))
        np.random.seed(seed)

        if method == "lhs":
            opts = filterkeys(kwargs, ['criterion', 'iterations'])
            xparams = self.sample_lhs(size, **opts)
        else:
            xparams = self.sample_montecarlo(size)
        return xparams

    def product(self):
        """only if all parameters are discrete
        """
        for p in self.params:
            if not isinstance(p.dist, DiscreteDist):
                raise TypeError("cannot make product of continuous distributions: "+p.name)

        pmatrix = list(product(*[p.dist.values for p in self.params]))
        return XParams(pmatrix, self.names)


    def filter_params(self, names, keep=True):
        if keep:
            self.params = [p for p in self.params if p.name in names]
        else:
            self.params = [p for p in self.params if p.name not in names]

    #TODO: `bounds` method for resampling?


    #def tojson(self, sort_keys=True, **kwargs):
    #    """Create json-compatible configuration file
    #    """
    #    cfg = {
    #        "params": [json.loads(p.tojson()) for p in self.params]
    #    }
    #    return json.dumps(cfg, sort_keys=True, **kwargs)

Prior = Params  # alias for back-compat

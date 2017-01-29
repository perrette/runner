"""Prior parameter sampling
"""
from __future__ import print_function, division
import json
from itertools import product
import sys
import numpy as np

import scipy.stats
from scipy.stats import norm, uniform

from simtools.tools import parse_dist, parse_list, parse_range, dist_to_str
from simtools.sampling.doelhs import lhs

import simtools.xparams as xp
from simtools.xparams import XParams

# default criterion for the lhs method
LHS_CRITERION = 'centermaximin' 

# for reading...
PRIOR_KEY = "prior"

class GenericParam(object):
    """scipy dist or discrete param
    """
    @staticmethod
    def parse(string):
        """Prior parameter defintion as NAME=SPEC.

        SPEC specifies param values or distribution.
        Discrete parameter values can be provided 
        as a comma-separated list `VALUE[,VALUE...]`
        or a range `START:STOP:N`.
        A distribution is provided as `TYPE?ARG,ARG[,ARG,...]`.
        Pre-defined `U?min,max` (uniform) and `N?mean,sd` (normal)
        or any scipy.stats distribution as TYPE?[SHP,]LOC,SCALE.")
        """
        # otherwise custom, command-line specific representation
        try:
            if '?' in string:
                param = PriorParam.parse(string)
            else:
                param = DiscreteParam.parse(string)

        except Exception as error:
            print( "ERROR:",error.message)
            raise
        return param

    @staticmethod
    def fromjson(string):
        if "values" in json.loads(string):
            return DiscreteParam.fromjson(string)
        else:
            return PriorParam.fromjson(string)

     
class PriorParam(GenericParam):
    """Prior parameter based on any scipy distribution
    """
    def __init__(self, name, dist, default=None):
        assert name, 'no param name'
        self.name = name
        self.dist = dist
        self.default = default

    def sample(self, size):
        """Monte Carlo sampling
        """
        return self.dist.rvs(size)

    def quantile(self, q):
        return self.dist.ppf(q)


    def __str__(self):
        return "{}={}".format(self.name, dist_to_str(self.dist))

    def tojson(self, sort_keys=True, **kwargs):
        """dict representation to write to config file
        """
        dname=self.dist.dist.name
        dargs=self.dist.args

        if dname == "uniform":
            loc, scale = dargs
            pdef = {
                "range": [loc, loc+scale],
                "dist": dname,
            }
        elif dname == "norm":
            loc, scale = dargs
            pdef = {
                "mean": loc,
                "std": scale,
                "dist": "normal",
            }
        else:
            pdef = {
                "dist": dname,
                "args": dargs,
            }

        pdef["name"] = self.name
        if self.default:
            pdef["default"] = self.default

        return json.dumps(pdef, sort_keys=sort_keys, **kwargs)


    @classmethod
    def fromjson(cls, string):
        """initialize from prior.json config (dat is a dict)
        """
        kw = json.loads(string)
        name = kw["name"]

        dname = kw.pop("dist", "uniform")
        args = kw.pop("args", None)

        if dname == "uniform":
            lo, hi = kw["range"]
            args = lo, hi-lo
        elif dname == "normal":
            dname = "norm"
            args = kw["mean"], kw["std"]
        elif not hasattr(scipy.stats.distributions, dname):
            raise ValueError("invalid distribution: "+dname)

        dist = getattr(scipy.stats.distributions, dname)
        return cls(name, dist(*args))


    @classmethod
    def parse(cls, string):
        """NAME=N?MEAN,STD or NAME=U?MIN,MAX or NAME=TYPE?ARG1[,ARG2 ...] where
        TYPE is any scipy.stats distribution with *shp, loc, scale parameters.
        """
        name, spec = string.split('=')
        if '!' in spec:
            spec, default = spec.split('!')
            default = parse_val(default)
        else:
            default = None
        dist = parse_dist(spec)
        return cls(name, dist, default)


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


class DiscreteParam(GenericParam):
    """Prior parameter that takes a number of discrete values
    """
    def __init__(self, name, values):
        self.name = name
        self.values = np.asarray(values)
    
    def sample(self, size):
        indices = np.random.randint(0, len(self.values), size)
        return self.values[indices]

    def quantile(self, q, interpolation='nearest'):
        return np.percentile(self.values, q*100, interpolation=interpolation)

    def __str__(self):
        " format in a similar way to what was provided as command-arg"
        args=",".join(*[str(v) for v in self.values])
        return "{}={}".format(self.name,args)

    @classmethod
    def parse(cls, string):
        name, spec = string.split("=")
        if ':' in spec:
            values = parse_range(spec)
        else:
            values = parse_list(spec)
        return cls(name, values)


    def tojson(self, sort_keys=True, **kwargs):
        return json.dumps({
            "name":self.name,
            "values":self.values.tolist(),
        }, sort_keys=sort_keys, **kwargs)

    @classmethod
    def fromjson(cls, string):
        kw = json.loads(string)
        return cls(kw["name"], kw["values"])



# json-compatible I/O
# ===================

def filterargs(kwargs, keys):
    """Only keep some of the keeps in a dictionary
    This is convenient for wrapper functions/methods, to avoid setting a default 
    parameter value at each level of dispatching.
    """
    return {k:kwargs[k] for k in kwargs if k in keys}



class Prior(object):
    def __init__(self, params):
        " list of PriorParam instances (for product)"
        self.params = list(params)
        for p in self.params:
            if not isinstance(p, GenericParam):
                raise TypeError("expected GenericParam, got:"+repr(type(p)))

    @classmethod
    def read(cls, file, key=PRIOR_KEY, param_cls=GenericParam):
        """read from config file

        file : json file
        key : sub-part of a larger json file?
        param_cls : optional, e.g. pick only PriorParam or DiscreteParam
            (for more informative error messages)
        """
        cfg = json.load(open(file))
        if key and key in cfg: cfg = cfg[key]
        params = [param_cls.fromjson(json.dumps(p)) for p in cfg["params"]]
        return cls(params)


    @property
    def names(self):
        return [p.name for p in self.params]

    def sample_montecarlo(self, size):
        """Basic montecarlo sampling --> return XParams
        """
        pmatrix = np.empty((size,len(self.names)))

        for i, p in enumerate(self.params):
            pmatrix[:,i] = p.sample(size=size) # scipy distribution: sample !

        return XParams(pmatrix, self.names)

    def sample_lhs(self, size, criterion=LHS_CRITERION, iterations=None):
        """Latin hypercube sampling --> return Xparams
        """
        #from pyDOE import lhs

        pmatrix = np.empty((size,len(self.names)))
        lhd = lhs(len(self.names), size, criterion, iterations) # sample x parameters, all in [0, 1]

        for i, p in enumerate(self.params):
            pmatrix[:,i] = p.quantile(lhd[:,i]) # take the quantile for the particular distribution

        return XParams(pmatrix, self.names)

    def sample(self, size, seed=None, method="lhs", **kwargs):
        """Wrapper for the various sampling methods. Unused **kwargs are ignored.
        """
        pmatrix = np.empty((size,len(self.names)))
        np.random.seed(seed)

        if method == "lhs":
            opts = filterargs(kwargs, ['criterion', 'iterations'])
            xparams = self.sample_lhs(size, **opts)
        else:
            xparams = self.sample_montecarlo(size)
        return xparams

    def product(self):
        """only if all parameters are discrete
        """
        for p in self.params:
            if not isinstance(p, DiscreteParam):
                raise TypeError("cannot make product of continuous distributions: "+p.name)

        pmatrix = list(product(*[p.values for p in self.params]))
        return XParams(pmatrix, self.names)


    def filter_params(self, names, keep=True):
        if keep:
            self.params = [p for p in self.params if p.name in names]
        else:
            self.params = [p for p in self.params if p.name not in names]

    #TODO: `bounds` method for resampling?


    def tojson(self, sort_keys=True, **kwargs):
        """Create json-compatible configuration file
        """
        cfg = {
            "params": [json.loads(p.tojson()) for p in self.params]
        }
        return json.dumps(cfg, sort_keys=True, **kwargs)


"""Prior parameter sampling
"""
from __future__ import print_function, division
import json
from itertools import product
import argparse
import numpy as np

import scipy.stats
from scipy.stats import norm, uniform

from simtools.tools import parse_dist, parse_list, parse_range, dist_to_str
from simtools.sampling.doelhs import lhs
from simtools.job.parsetools import Command, Job

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
    def fromdict(**kwargs):
        if "values" in kwargs:
            return DiscreteParam.fromdict(**kwargs)
        else:
            return PriorParam.fromdict(**kwargs)

     
class PriorParam(GenericParam):
    """Prior parameter based on any scipy distribution
    """
    def __init__(self, name, dist):
        self.name = name
        self.dist = dist

    def sample(self, size):
        """Monte Carlo sampling
        """
        return self.dist.rvs(size)

    def quantile(self, q):
        return self.dist.ppf(q)


    def __str__(self):
        return "{}={}".format(self.name, dist_to_str(self.dist))


    def todict(self):
        """dict representation to write to config file
        """
        dname=self.dist.dist.name
        dargs=self.dist.args

        if dname == "uniform":
            loc, scale = dargs
            pdef = {
                "range": [loc, loc+scale],
            }
        elif dname == "norm":
            loc, scale = dargs
            pdef = {
                "mean": loc,
                "std": scale,
            }
        else:
            pdef = {
                "dist": dname,
                "args": dargs,
            }

        pdef["name"] = self.name

        return pdef


    @classmethod
    def fromdict(cls, **kw):
        """initialize from prior.json config (dat is a dict)
        """
        name = kw["name"]

        dname = kw.pop("dist", None)
        args = kw.pop("args", None)

        if not dname:
            if "range" in kw:
                dname = "uniform"
                lo, hi = kw["range"]
                args = lo, hi-lo
            elif "mean" in kw:
                dname = "norm"
                args = kw["mean"], kw["std"]
            else:
                raise ValueError("invalid distribution")

        dist = getattr(scipy.stats.distributions, dname)
        return cls(name, dist(*args))


    @classmethod
    def parse(cls, string):
        name, spec = string.split('=')
        dist = parse_dist(spec)
        return cls(name, dist)


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


    def todict(self):
        return {
            "name":self.name,
            "values":self.values.tolist(),
        }

    @classmethod
    def fromdict(cls, **kw):
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
            if not isinstance(p, PriorParam):
                raise TypeError(repr(p))

    @classmethod
    def read(cls, file, key=PRIOR_KEY, param_cls=GenericParam):
        """read from config file

        file : json file
        key : sub-part of a larger json file?
        param_cls : optional, e.g. pick only PriorParam or DiscreteParam
            (for more informative error messages)
        """
        cfg = json.load(open(file))
        if key: cfg = cfg[key]
        params = [param_cls.fromdict(**p) for p in cfg["params"]]
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

    #TODO: `bounds` method for resampling


########################################################################
#
# 
#
class PriorParser(object):

    @staticmethod
    def add_arguments(parser, file_required=False, root=PRIOR_KEY):
        """
        parser : argparser.ArgumentParser instance
        returns the class constructor from_parser_namespace
        """
        grp = parser.add_argument_group("prior parameters")
        grp.add_argument('-p', '--prior-params', default=[], nargs='*', 
                                type=GenericParam.parse, metavar="NAME=SPEC", 
                                help=GenericParam.parse.__doc__)

        grp.add_argument('--config', required=file_required,
                         help='input prior parameter file (json file with "'+root+'" key)')

        grp.add_argument('--prior-key', default=root, help=argparse.SUPPRESS)

        x = grp.add_mutually_exclusive_group()
        x.add_argument('--only-params', nargs='*', 
                         help="filter out all but these parameters")
        x.add_argument('--exclude-params', nargs='*', 
                         help="filter out these parameters")
        grp.add_argument("--add", nargs='*', type=GenericParam.parse)

    @staticmethod
    def from_namespace(args):
        """return Prior class
        """
        if args.config:
            prior = Prior.read(args.config, args.prior_key)
            if args.only_params:
                prior.filter_params(args.only_params, keep=True)
            if args.exclude_params:
                prior.filter_params(args.exclude_params, keep=False)

        else:
            prior = Prior(args.prior_params)

        return prior


class EditPriorConfig(Command):
    """edit config w.r.t Prior Params and print to stdout
    """
    def __init__(self, parser):

        PriorParser.add_arguments(parser, file_required=True)
        parser.add_argument("--full", action='store_true')


    def __call__(self, args):
        prior = Prior.read(args.config, args.prior_key)
        print(prior.params)
        print([p.todict() for p in args.add])
        if args.only_params:
            prior.filter_params(args.only_params, keep=True)
        if args.exclude_params:
            prior.filter_params(args.exclude_params, keep=False)

        cfg = {
            "params": [p.todict() for p in prior.params]
        }

        if args.full:
            full = json.load(open(args.config))
            full["prior"] = cfg
            cfg = full

        print(json.dumps(cfg, indent=2))


def main():
    EditPriorConfig.main()


if __name__ == "__main__":
    main()

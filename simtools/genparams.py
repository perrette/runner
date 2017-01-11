#!/usr/bin/env python2.7
"""Generate parameter ensemble
"""
from __future__ import print_function
import argparse, os
import json
from argparse import RawDescriptionHelpFormatter
from itertools import product
import numpy as np
from simtools.tools import DataFrame, parse_val
from simtools.resample import Resampler

# parse parameters from command-line
# ==================================
def parse_param_list(string):
    """Parse list of parameters VALUE[,VALUE,...]
    """
    return [parse_val(value) for value in string.split(',')]

def parse_param_range(string):
    """Parse parameters START:STOP:STEP
    """
    import numpy as np
    return np.arange(*[parse_val(value) for value in string.split(':')]).tolist()

def parse_param_dist(string):
    """Parse parameters dist?loc,scale
    """
    import scipy.stats.distributions as sd
    name,params = string.split('?')
    loc, scale = params.split(',')
    return getattr(sd,name)(parse_val(loc), parse_val(scale))


def params_parser(string):
    """used as type by ArgumentParser
    """
    try:
        name, spec = string.split('=')
        #if '?' in spec:
        #    params = parse_param_dist(spec)
        if ':' in spec:
            params = parse_param_range(spec)
        else:
            params = parse_param_list(spec)
    except Exception as error:
        print( "ERROR:",error.message)
        raise
    return name,params


class PriorParam(object):
    def __init__(self, name, dist):
        self.name = name
        self.dist = dist

    @classmethod
    def parse(cls, string):
        "NAME=SPEC"
        name, spec = string.split("=")
        dist = parse_param_dist(spec)
        return cls(name, dist)

    @classmethod
    def fromconfig(cls, dat):
        """initialize from prior.json config (dat is a dict)
        """
        from scipy.stats.distributions import uniform
        name = dat["name"]
        lo, hi = dat["range"]
        return cls(name, uniform(lo, hi-lo))


class PriorParams(object):
    def __init__(self, params):
        " list of PriorParam instances "
        self.params = params

    @classmethod
    def read(cls, file):
        """read from config file
        """
        dat = json.load(open(file))
        return cls([PriorParam.fromconfig(p) for p in dat["params"]])

    @property
    def names(self):
        return [p.name for p in self.params]

    def update(self, params, append=True):
        """Update existing prior parameter or append new ones.
        """
        for p in params:
            found = False
            for p0 in self.params:
                if p0.name == p.name:
                    p0.dist = p.dist
                    found = True
            if not found:
                if not append:
                    print("Existing parameters:", self.names)
                    raise ValueError(p.name+" not found.")
                self.params.append(p)

    def sample_montecarlo(self, size):
        """Basic montecarlo sampling --> return XParams
        """
        import numpy as np
        pmatrix = np.empty((size,len(self.names)))

        for i, p in enumerate(self.params):
            pmatrix[:,i] = p.dist.rvs(size=size) # scipy distribution: sample !

        return XParams(self.names, pmatrix)

    def sample_lhs(self, size, criterion=None, iterations=None):
        """Latin hypercube sampling --> return Xparams
        """
        import numpy as np
        from pyDOE import lhs

        pmatrix = np.empty((size,len(self.names)))
        lhd = lhs(len(self.names), size, criterion, iterations) # sample x parameters, all in [0, 1]

        for i, p in enumerate(self.params):
            pmatrix[:,i] = p.dist.ppf(lhd[:,i]) # take the percentiles for the particular distribution

        return XParams(pmatrix, self.names)

    def sample(self, size, seed=None, method="lhs", **kwargs):
        np.random.seed(seed)
        if method == "lhs":
            return self.sample_lhs(size, **kwargs)
        else:
            return self.sample_montecarlo(size, **kwargs)


class XParams(DataFrame):
    """Experiment params
    """
    def resample(self, weights, epsilon=None, **kwargs):
        resampler = Resampler(weights)
        vals = resampler.iis(self.values, epsilon, **kwargs)
        return XParams(vals, self.names)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
            epilog='Examples: \n ./genparams.py product -p a=0,2 b=0:3:1 c=4 \n ./genparams.py sample -p a=uniform?0,10 b=norm?0,2 --mode lhs --size 4',
            formatter_class=RawDescriptionHelpFormatter)

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('-o', '--out', help="Output parameter file")

    subparsers = parser.add_subparsers(dest="cmd")

    subp = subparsers.add_parser("product", parents=[parent],
                                 help="factorial combination of parameter values")
    subp.add_argument('-p', '--params', default=[], type=params_parser, nargs='*', 
                      metavar="NAME=SPEC", 
                      help="SPEC is a comma-separated list: VALUE[,VALUE...] OR a range: START:STOP:STEP")
    subp.add_argument('-i', '--params-file')

    subp = subparsers.add_parser("sample", parents=[parent], help='montecarlo sampling')
    subp.add_argument('-p', '--params', default=[], type=PriorParam.parse, nargs='*', 
                      metavar="NAME=SPEC",
            help="Modified parameters. SPEC is a scipy dist TYPE?LOC,SCALE, e.g. norm?mean,sd or uniform?min,max-min")
    subp.add_argument('-i', '--params-file')

    subp.add_argument('--mode', choices=['montecarlo','lhs'], 
                      default='montecarlo', 
                      help="Sampling mode: Monte Carlo or Latin Hypercube Sampling (default=%(default)s)")
    subp.add_argument('--lhs-criterion', help="pyDOE lhs parameter")
    subp.add_argument('--lhs-iterations', help="pyDOE lhs parameter")
    #parser.add_argument('-i','--from-file', help="look in file for any parameter provided as params, and use instead of command-line specification")
    subp.add_argument('-N', '--size',type=int, help="Sample size (montecarlo or lhs modes)")
    subp.add_argument('--seed', type=int, 
                      help="random seed, for reproducible results (default to None)")
    return parser

def main(argv=None):

    parser = get_parser()
    args = parser.parse_args(argv)


    # Combine parameter values
    # ...factorial model: no numpy distribution allowed
    if args.cmd == 'product':
        pnames = [nm for nm, vals in args.params]
        pmatrix = list(product(*[vals for nm, vals in args.params]))
        xparams = XParams(pmatrx, pnames)
        

    # ...monte carlo and lhs mode
    elif args.cmd == 'sample':

        if args.params_file:
            prior = PriorParams.read(args.params_file)
            prior.update(args.params)
        else:
            prior = PriorParams(args.params)

        assert args.size is not None, "need to provide --size for sampling"

        if args.mode == "lhs":
            xparams = prior.sample_lhs(args.size, seed=args.seed, 
                                       criterion=args.lhs_criterion, 
                                       iterations=args.lhs_iterations)
        else:
            xparams = prior.sample(args.size, seed=args.seed)

    if args.out:
        with open(args.out,'w') as f:
            f.write(str(xparams))
    else:
        print (str(xparams))


if __name__ == '__main__':
    main()

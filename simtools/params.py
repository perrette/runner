"""Generate parameter ensemble
"""
from __future__ import print_function, division
import json
from itertools import product
import argparse
import numpy as np
from simtools.tools import parse_val, DataFrame
from simtools.sampling.resampling import Resampler, RESAMPLING_METHOD, NEFF_BOUNDS
from simtools.sampling.doelhs import lhs



# default criterion for the lhs method
LHS_CRITERION = 'centermaximin' 

def filterargs(kwargs, keys):
    """Only keep some of the keeps in a dictionary
    This is convenient for wrapper functions/methods, to avoid setting a default 
    parameter value at each level of dispatching.
    """
    return {k:kwargs[k] for k in kwargs if k in keys}


class PriorParam(object):
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


class DiscreteParam(PriorParam):
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


class PriorParams(object):
    def __init__(self, params):
        " list of PriorParam instances "
        self.params = list(params)
        for p in self.params:
            if not isinstance(p, PriorParam):
                raise TypeError(repr(p))

    @classmethod
    def read(cls, file):
        """read from config file
        """
        dat = json.load(open(file))
        return cls([fromconfig(p) for p in dat["params"]])

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


class XParams(DataFrame):
    """Experiment params
    """
    def __init__(self, values, names, default=None):
        self.values = values 
        self.names = names
        self.default = default

    def pset_as_array(self, i=None):
        if i is None:
            pvalues = self.default
        else:
            pvalues = self.values[i]
        return pvalues

    def pset_as_dict(self, i=None):
        """return parameter set as a dictionary
        """
        pvalues = self.pset_as_array(i)

        if pvalues is None:
            return {}  # case were default parameters are not provided

        params = {}
        for k, v in zip(self.names, pvalues):
            params[k] = v
        return params

    def resample(self, weights, size=None, seed=None, method=RESAMPLING_METHOD, 
                 iis=False, epsilon=None, neff_bounds=NEFF_BOUNDS, bounds=None):
        """
        Parameters
        ----------
        weights : array of weights (must match params' size)
        size : new ensemble size, by default same as current
        seed : random state seed (None)
        method : method for weighted resampling (see simtools.resample.Resampler)
        iis : step of the Iterative Importance Sampling strategy (Hannan and Hargreave)
            where weights are flattened (epsilon exponent) and jitter (noise) is added
            to the resampled ensemble, as a fraction epsilon of its (weighted) 
            covariance. In the linear case, the combination of flattened resampling
            and jitter addition is equivalent to one time resampling with full weights.
        epsilon : scaling exponent for the weights, ie `weights**epsilon` [iis method only] 
            If not provided, epsilon is automatically generated to yield an effective
            ensemble size comprised in the neff_bounds range. Starting value: epsilon.
        neff_bounds : target effective ensemble size to determine epsilon automatically
        bounds : authorized parameter range (experimental). If jitter addition yields parameters
            outside the specified range, try again a number of times. [iis method only]


        Returns
        -------
        XParams instance
        """
        if weights.size != self.size:
            raise ValueError("params and weights size do not match")

        resampler = Resampler(weights) # default size implied by weights
        if iis:
            vals = resampler.iis(self.values, 
                           size=size, seed=seed, method=method, 
                           bounds=bounds, neff_bounds=neff_bounds, 
                           epsilon=epsilon)

        else:
            idx = resampler.sample(size=size, seed=seed, method=method)
            vals = self.values[idx]
        return XParams(vals, self.names)



# Functions to help command-line and interactivity
# ================================================

# parse parameters from command-line
# ----------------------------------
def parse_param_list(string):
    """Parse list of parameters VALUE[,VALUE,...]
    """
    return [parse_val(value) for value in string.split(',')]

def parse_param_range(string):
    """Parse parameters START:STOP:N
    """
    start, stop, n = string.split(':')
    start = float(start)
    stop = float(stop)
    n = int(n)
    return np.linspace(start, stop, n).tolist()

def parse_param_dist(string):
    """Parse parameters dist?loc,scale
    """
    import scipy.stats.distributions as sd
    name,spec = string.split('?')
    args = [float(a) for a in spec.split(',')]
    
    # alias for common cases
    if name == "N":
        mean, std = args
        dist = sd.norm(mean, std)

    elif name == "U":
        lo, hi = args  # note: uniform?loc,scale differs !
        dist = sd.uniform(lo, hi-lo) 

    else:
        dist = getattr(sd,name)(*args)

    return dist

def parse_param(string):
    """used as type by ArgumentParser
    """
    try:
        name, spec = string.split('=')
        if '?' in spec:
            dist = parse_param_dist(spec)
            param = PriorParam(name, dist)
        elif ':' in spec:
            values = parse_param_range(spec)
            param = DiscreteParam(name, values)
        else:
            values = parse_param_list(spec)
            param = DiscreteParam(name, values)
    except Exception as error:
        print( "ERROR:",error.message)
        raise
    return param


def fromconfig(dat):
    """initialize from prior.json config (dat is a dict)
    """
    from scipy.stats.distributions import uniform
    name = dat["name"]
    lo, hi = dat["range"]
    return PriorParam(name, uniform(lo, hi-lo))


# for the sake of the scripts...
def update_params(params, pp, append=True):
    """Update existing prior parameter (by name) or append new ones.

    params, pp : list of PriorParam classes
    """
    for p in pp:
        found = False
        for i, p0 in enumerate(params):
            if p0.name == p.name:
                params[i] = p
                found = True
        if not found:
            if not append:
                print("Existing parameters:", ", ".join([p0.name for p0 in params]))
                raise ValueError(p.name+" not found.")
            params.append(p)


class ParamsParser(object):
    """Helper class to build ArgumentParser with subcommand and keep clean
    """
    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.subparsers = self.parser.add_subparsers(dest='cmd')
        self.define_parents()

    def define_parents(self):
        " arguments shared in various commands "
        # input prior parameters
        self.prior = argparse.ArgumentParser(add_help=False)
        self.prior.add_argument('-p', '--prior-params', default=[], nargs='*', 
                                type=parse_param, metavar="NAME=SPEC", 
                                help="Prior parameter defintion. \
SPEC specifies a param values or distribution (depending on the sub-command).\
            Discrete parameter values can be provided \
            as a comma-separated list `VALUE[,VALUE...]` \
            or a range `START:STOP:N`. \
            A distribution is provided as `TYPE?ARG,ARG[,ARG,...]`. \
            Pre-defined `U?min,max` (uniform) and `N?mean,sd` (normal) \
            or any scipy.stats distribution as TYPE?[SHP,]LOC,SCALE.")

        self.prior.add_argument('--prior-file', help='experimental')

        ## size & sample
        #self.size = argparse.ArgumentParser(add_help=False)

        # input param file
        self.pin = argparse.ArgumentParser(add_help=False)
        self.pin.add_argument('params-file', help='input parameter file')

        # input ensemble weights
        self.win = argparse.ArgumentParser(add_help=False)
        group = self.win.add_argument_group('weights')
        group.add_argument('-w','--weights-file', required=True)
        group.add_argument('--log', action='store_true', 
                           help='weights are provided as log-likelihood?')

        # output param file
        self.pout = argparse.ArgumentParser(add_help=False)
        self.pout.add_argument('-o', '--out', help="output parameter file")


    def add_product(self):
        """factorial combination of parameter values
        """
        subp = self.subparsers.add_parser("product", parents=[self.prior, self.pout],
                                     help=self.add_product.__doc__)

    def add_sample(self):
        """Sample prior parameter distribution
        """
        subp = self.subparsers.add_parser("sample", parents=[self.prior, self.pout], 
                                     help=__doc__)

        subp.add_argument('-N', '--size',type=int, required=True, 
                          help="Sample size")
        subp.add_argument('--seed', type=int, 
                          help="random seed, for reproducible results (default to None)")
        subp.add_argument('--method', choices=['montecarlo','lhs'], 
                          default='lhs', 
                          help="Sampling method: Monte Carlo or Latin Hypercube Sampling (default=%(default)s)")

        grp = subp.add_argument_group('Latin Hypercube Sampling (pyDOE)')
        grp.add_argument('--lhs-criterion', default=LHS_CRITERION,
                          help="see pyDOE.lhs (default=%(default)s)")
        grp.add_argument('--lhs-iterations', type=int, help="see pyDOE.lhs")


    def add_resample(self):
        """Resample an existing parameter set using weights.
        """
        subp = self.subparsers.add_parser("resample", 
                                          parents=[self.pout, self.pin, self.win], 
                                     help=__doc__)

        subp.add_argument('-N', '--size', help="New sample size (default: same size as before)", type=int)
        subp.add_argument('--seed', type=int, help="random seed, for reproducible results (default to None)")

        group = subp.add_argument_group('iis')
        group.add_argument('--iis', action='store_true', 
                          help="IIS-type resampling with likeihood flattening + jitter")
        group.add_argument('--epsilon', type=float, 
                           help='Exponent to flatten the weights and derive jitter \
variance as a fraction of resampled parameter variance. \
            If not provided 0.05 is used as a starting value but adjusted if the \
        effective ensemble size is not in the range specified by --neff-bounds.')

        group.add_argument('--neff-bounds', nargs=2, default=NEFF_BOUNDS, type=int, 
                           help='Acceptable range for the effective ensemble size\
                           when --epsilon is not provided. Default to %(default)s.')

        group = subp.add_argument_group('sampling')
        group.add_argument('--method', choices=['residual', 'multinomial'], 
                           default=RESAMPLING_METHOD, 
                           help='resampling method (default: %(default)s)')

    def add_neff(self):
        """Check effective ensemble size
        """
        subp = self.subparsers.add_parser("neff", parents=[self.pin, self.win], 
                                     help=__doc__)
        subp.add_argument('--epsilon', type=float, default=1, 
                          help='likelihood flattening, see resample sub-command')

        #group.add_argument('--neff-only', action='store_true', help='effective ensemble size')

    def parse_args(self, *args, **kwargs):
        return self.parser.parse_args(*args, **kwargs)


def main(argv=None):

    parser = ParamsParser(description=__doc__,
            epilog='Examples: \n ./genparams.py product -p a=0,2 b=0:3:1 c=4 \n ./genparams.py sample -p a=uniform?0,10 b=norm?0,2 --method lhs --size 4',
            formatter_class=argparse.RawDescriptionHelpFormatter)

    # add subcommands
    parser.add_product()
    parser.add_sample()
    parser.add_resample()
    parser.add_neff()

    args = parser.parse_args(argv)

    if args.prior_file:
        prior = PriorParams.read(args.prior_file)
        update_params(prior.params, args.prior_params)
    else:
        prior = PriorParams(args.prior_params)


    # Combine parameter values
    # ...factorial model: no numpy distribution allowed
    if args.cmd == 'product':
        xparams = prior.product()

    # ...monte carlo and lhs mode
    elif args.cmd == 'sample':
        xparams = prior.sample(args.size, seed=args.seed, 
                               method=args.method,
                               criterion=args.lhs_criterion,
                               iterations=args.lhs_iterations)

    elif args.cmd == 'resample':

        xpin = XParams.read(args.params_file)
        w = np.loadtxt(args.weights_file)
        if args.log:
            w = np.exp(log)
        xparams = xpin.resample(w, size=args.size, seed=args.seed,
                                method=args.method,
                                iis=args.iis, epsilon=args.epsilon, 
                                neff_bounds=args.neff_bounds, 
                                )

    elif args.cmd == 'neff':
        w = np.loadtxt(args.weights_file)
        if args.log:
            w = np.exp(log * args.epsilon)
        else:
            w = w ** args.epsilon
        print( Resampler(w).neff() )
        return


    if args.out:
        with open(args.out,'w') as f:
            f.write(str(xparams))
    else:
        print (str(xparams))

"""Prior parameter sampling
"""
from __future__ import print_function, division
import json
from itertools import product
import argparse
import sys
import numpy as np

import scipy.stats
from scipy.stats import norm, uniform

from simtools.tools import parse_dist, parse_list, parse_range, dist_to_str
from simtools.sampling.doelhs import lhs
from simtools.parsetools import ObjectParser, ProgramParser, Job

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
        # first try json format
        try:
            return GenericParam._fromjson(string)
        except:
            pass

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
    def _fromjson(string):
        if "values" in json.loads(string):
            return DiscreteParam._fromjson(string)
        else:
            return PriorParam._fromjson(string)

     
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

    def tojson(self, sort_keys=True, **kwargs):
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

        return json.dumps(pdef, sort_keys=sort_keys, **kwargs)


    @classmethod
    def _fromjson(cls, string):
        """initialize from prior.json config (dat is a dict)
        """
        kw = json.loads(string)
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
        try:
            return cls._fromjson(string)
        except:
            pass
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
        try:
            return cls._fromjson(string)
        except:
            pass
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
    def _fromjson(cls, string):
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
        params = [param_cls.parse(json.dumps(p)) for p in cfg["params"]]
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
class PriorParser(ObjectParser):
    """
    """
    def __init__(self, file_required=False):
        self.file_required = file_required

    def add_arguments(self, parser):
        """
        parser : argparser.ArgumentParser instance
        returns the class constructor from_parser_namespace
        """
        grp = parser.add_argument_group("prior parameters")
        grp.add_argument('-p', '--prior-params', default=[], nargs='*', 
                                type=GenericParam.parse, metavar="NAME=SPEC", 
                                help=GenericParam.parse.__doc__)

        grp.add_argument('--config', required=self.file_required,
                         help='input config file')

        x = grp.add_mutually_exclusive_group()
        x.add_argument('--only-params', nargs='*', 
                         help="filter out all but these parameters")
        x.add_argument('--exclude-params', nargs='*', 
                         help="filter out these parameters")


    def postprocess(self, args):
        """return Prior class from namespace
        """
        if args.config:
            prior = Prior.read(args.config)
            update = Prior(args.prior_params)
            for p in update.params:
                try:
                    i = prior.names.index(p.name)
                    prior.params[i] = p
                except ValueError:
                    prior.params.append(p)

        else:
            prior = Prior(args.prior_params)

        if args.only_params:
            prior.filter_params(args.only_params, keep=True)
        if args.exclude_params:
            prior.filter_params(args.exclude_params, keep=False)

        return prior



class WeightsParser(ObjectParser):
    def __init__(self, required=True):
        self.required = required

    def add_arguments(self, parser):
        group = parser.add_argument_group('weights')
        group.add_argument('-w','--weights-file', required=self.required)
        group.add_argument('--log', action='store_true', 
                           help='weights are provided as log-likelihood?')

    def postprocess(self, args):
        w = np.loadtxt(args.weights_file)
        if args.log:
            w = np.exp(log)
        return w


# Programs
# ========
def show_config_prog(argv=None):
    """edit config w.r.t Prior Params and print to stdout
    """
    prog = ProgramParser(description=main.__doc__)
    prior_parser = PriorParser()
    prog.add_object_parser(prior_parser, 'prior')

    prog.add_argument("--full", action='store_true')
    prog.add_argument("--indent", type=int)
    args = prog.parse_objects(argv)

    cfg = {
        "params": [json.loads(p.tojson()) for p in args.prior.params]
    }

    if args.full:
        if not args.config:
            print("ERROR: `--config FILE` must be provided with the `--full` option")
            sys.exit(1)
        full = json.load(open(args.config))
        full["prior"] = cfg
        cfg = full

    print(json.dumps(cfg, indent=args.indent, sort_keys=True))


def return_params(xparams, out):
    if out:
        with open(out, "w") as f:
            f.write(str(xparams))
    else:
        print(str(xparams))


def product_prog(argv=None):
    """Factorial combination of parameter values
    """
    prog = ProgramParser()
    prog.add_object_parser(PriorParser(), 'prior')
    prog.add_argument('-o', '--out', help="output parameter file")

    o = prog.parse_objects(argv)

    xparams = o.prior.product()
    return return_params(xparams, o.out)


def sample_prog(argv=None):
    """Sample prior parameter distribution
    """
    prog = ProgramParser()
    prog.add_object_parser(PriorParser(), 'prior')
    prog.add_argument('-o', '--out', help="output parameter file")

    prog.add_argument('-N', '--size',type=int, required=True, 
                      help="Sample size")
    prog.add_argument('--seed', type=int, 
                      help="random seed, for reproducible results (default to None)")
    prog.add_argument('--method', choices=['montecarlo','lhs'], 
                      default='lhs', 
                      help="Sampling method: Monte Carlo or Latin Hypercube Sampling (default=%(default)s)")

    grp = prog.add_argument_group('Latin Hypercube Sampling (pyDOE)')
    grp.add_argument('--lhs-criterion', default=LHS_CRITERION,
                      help="see pyDOE.lhs (default=%(default)s)")
    grp.add_argument('--lhs-iterations', type=int, help="see pyDOE.lhs")


    o = prog.parse_objects(argv)

    xparams = o.prior.sample(o.size, seed=o.seed, 
                           method=o.method,
                           criterion=o.lhs_criterion,
                           iterations=o.lhs_iterations)

    return return_params(xparams, o.out)



def resample_prog(argv=None):
    """Resample an existing parameter set using weights.
    """
    prog = ProgramParser(description=resample_prog.__doc__)
    prog.add_object_parser(PriorParser(), 'prior')

    #PriorParser.add_argument(parser)
    # TODO: parameterize this argument to make it POSITIONAL (FunWrapper)
    prog.add_argument("params_file", 
                        help="ensemble parameter flle to resample")

    prog.add_object_parser(WeightsParser(), "weights")


    prog.add_argument('-N', '--size', help="New sample size (default: same size as before)", type=int)
    prog.add_argument('--seed', type=int, help="random seed, for reproducible results (default to None)")

    group = prog.add_argument_group('iis')
    group.add_argument('--iis', action='store_true', 
                      help="IIS-type resampling with likeihood flattening + jitter")
    group.add_argument('--epsilon', type=float, 
                       help='Exponent to flatten the weights and derive jitter \
variance as a fraction of resampled parameter variance. \
        If not provided 0.05 is used as a starting value but adjusted if the \
    effective ensemble size is not in the range specified by --neff-bounds.')

    group.add_argument('--neff-bounds', nargs=2, default=xp.NEFF_BOUNDS, type=int, 
                       help='Acceptable range for the effective ensemble size\
                       when --epsilon is not provided. Default to %(default)s.')

    group = prog.add_argument_group('sampling')
    group.add_argument('--method', choices=['residual', 'multinomial'], 
                       default=xp.RESAMPLING_METHOD, 
                       help='resampling method (default: %(default)s)')


    args = prog.parse_objects(argv)
    xpin = XParams.read(args.params_file)
    xparams = xpin.resample(args.weights, size=args.size, seed=args.seed,
                            method=args.method,
                            iis=args.iis, epsilon=args.epsilon, 
                            neff_bounds=args.neff_bounds, 
                            )
    return return_params(xparams, args)


def neff_prog(argv=None):
    """Check effective ensemble size
    """
    prog = ProgramParser()
    prog.add_object_parser(WeightsParser(), "weights")
    prog.add_argument('--epsilon', type=float, default=1, 
                      help='likelihood flattening, see resample sub-command')

    print( Resampler(args.weights**args.epsilon).neff() )


def main(argv=None):

    job = Job()
    job.add_command("show", show_config_prog)
    job.add_command("product", product_prog)
    job.add_command("sample", sample_prog)
    job.add_command("resample", resample_prog)
    job.add_command("neff", neff_prog)
    #job.add_command("neff", Neff)

    job.main(argv)


if __name__ == "__main__":
    main()

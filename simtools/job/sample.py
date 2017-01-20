"""Jobs to generate parameters for the ensemble (sampling, resampling..
"""
from __future__ import print_function, division
import json
import inspect
import argparse
import sys
import numpy as np

from simtools.parsetools import CustomParser, Job
from simtools.prior import Prior, GenericParam, LHS_CRITERION

# Get prior parameter
# ===================
def getprior(prior_params=None, config_file=None, only_params=None, exclude_params=None):
    """Return prior parameters from configuration file and any command-line update

    * prior_params : [GenericParam]
    * config_file : config file to define params from
    * only_params : [str] filter out all but these parameters
    * exclude_params : [str] filter out these parameters
    """
    if config_file:
        prior = Prior.read(config_file)
        update = Prior(prior_params or [])
        for p in update.params:
            try:
                i = prior.names.index(p.name)
                prior.params[i] = p
            except ValueError:
                prior.params.append(p)

    else:
        prior = Prior(prior_params or [])

    if only_params:
        prior.filter_params(only_params, keep=True)
    if exclude_params:
        prior.filter_params(exclude_params, keep=False)

    if not prior.params:
        prior_parser.error("either --prior-params or --config-file must be provided")
        sys.exit(1) 

    return prior


prior_parser = CustomParser(add_help=False)
grp = prior_parser.add_argument_group('prior parameters')
grp.add_argument('--prior-params', '-p',
                         type=GenericParam.parse,
                         help=GenericParam.parse.__doc__,
                         metavar="NAME=SPEC",
                         nargs='*',
                         default = [])
#prior_parser.add_argument('--prior-file', dest="config_file", help=argparse.SUPPRESS)
grp.add_argument('--config-file', dest="config_file", help="configuration file")

x = grp.add_mutually_exclusive_group()
x.add_argument('--only-params', nargs='*', 
                 help="filter out all but these parameters")
x.add_argument('--exclude-params', nargs='*', 
                 help="filter out these parameters")

prior_parser.add_postprocessor(getprior, inspect=True, dest='prior')



# Return new ensemble parameters
# ------------------------------

def return_params(xparams, out):
    if out:
        with open(out, "w") as f:
            f.write(str(xparams))
    else:
        print(str(xparams))


def product_main(argv=None):
    """Factorial combination of parameter values
    """
    parser = CustomParser(description=product_main.__doc__, parents=[prior_parser])
    parser.add_argument('-o', '--out', help="output parameter file")
    o = parser.parse_args(argv)
    o = parser.postprocess(o) # add prior
    xparams = o.prior.product()
    return return_params(xparams, o.out)


def sample_main(argv=None):
    """Sample prior parameter distribution
    """
    parser = CustomParser(description=sample_main.__doc__, parents=[prior_parser])
    parser.add_argument('-o', '--out', help="output parameter file")

    parser.add_argument('-N', '--size',type=int, required=True, 
                      help="Sample size")
    parser.add_argument('--seed', type=int, 
                      help="random seed, for reproducible results (default to None)")
    parser.add_argument('--method', choices=['montecarlo','lhs'], 
                      default='lhs', 
                      help="Sampling method: Monte Carlo or Latin Hypercube Sampling (default=%(default)s)")

    grp = parser.add_argument_group('Latin Hypercube Sampling (pyDOE)')
    grp.add_argument('--lhs-criterion', default=LHS_CRITERION,
                      help="see pyDOE.lhs (default=%(default)s)")
    grp.add_argument('--lhs-iterations', type=int, help="see pyDOE.lhs")


    o = parser.parse_args(argv)
    o = parser.postprocess(o)

    xparams = o.prior.sample(o.size, seed=o.seed, 
                           method=o.method,
                           criterion=o.lhs_criterion,
                           iterations=o.lhs_iterations)

    return return_params(xparams, o.out)


def prior_config_main(argv=None):
    """update prior config file and print result to screen
    """
    parser = CustomParser(description="update prior config file and print result to screen", 
                          parents=[prior_parser])
    parser.add_argument("--full", 
                        action='store_true', 
                        help='show full configuration file (including model etc.)')
    parser.add_argument("--indent", type=int)

    o = parser.parse_args(argv)
    o = parser.postprocess(o)

    jsonstring = o.prior.tojson(indent=o.indent)
    if o.full:
        if not o.config_file:
            parser.error("The `--full` option requires `--config-file FILE` to be provided")
        cfg = json.load(open(o.config_file))
        cfg.update(json.loads(jsonstring))
        jsonstring = json.dumps(cfg, sort_keys=True, indent=o.indent)

    print(jsonstring)

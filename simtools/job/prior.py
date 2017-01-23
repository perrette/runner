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
from simtools.job.config import prior_parser


# Return new ensemble parameters
# ------------------------------

def return_params(xparams, out):
    if out:
        with open(out, "w") as f:
            f.write(str(xparams))
    else:
        print(str(xparams))


def product(argv=None):
    """Factorial combination of parameter values
    """
    parser = CustomParser(description=product.__doc__, parents=[prior_parser])
    parser.add_argument('-o', '--out', help="output parameter file")
    o = parser.parse_args(argv)
    o = parser.postprocess(o) # add prior
    xparams = o.prior.product()
    return return_params(xparams, o.out)


def sample(argv=None):
    """Sample prior parameter distribution
    """
    parser = CustomParser(description=sample.__doc__, parents=[prior_parser])
    #grp.add_argument('--config-file', dest="config_file", help="configuration file")
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

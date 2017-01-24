"""Jobs to generate parameters for the ensemble (sampling, resampling..
"""
from __future__ import print_function, division
import json
import inspect
import argparse
import sys
import numpy as np

from simtools.prior import Prior, GenericParam, LHS_CRITERION
#from simtools.job.tools import CustomParser
#from simtools.job.config import prior_parser
from simtools.job.config import Program, PriorConfig, SubConfig


# Return new ensemble parameters
# ------------------------------
def return_params(xparams, out):
    if out:
        with open(out, "w") as f:
            f.write(str(xparams))
    else:
        print(str(xparams))


#TODO: check why inverting super class results in bug
#class Product(PriorConfig, Program): 
class Product(Program, PriorConfig):
    """Factorial combination of parameter values

    * out : "output parameter file"
    """
    def __init__(self, out=None, **kwargs):
        self.out = out
        self._super_init(self, **kwargs)

    @property
    def parser(self):
        return self._parser_auto(add_help=True)

    def main(self):
        xparams = self.getprior().product()
        return return_params(xparams, self.out)

product = Product()

class LatinHyperCube(SubConfig):
    """Latin Hypercube Sampling (pyDOE)

    * lhs_criterion : see pyDOE.lhs
    * lhs_iterations : see pyDOE.lhs
    """
    def __init__(self, lhs_criterion=LHS_CRITERION, lhs_iterations=None):
        self.lhs_criterion = lhs_criterion
        self.lhs_iterations = lhs_iterations

    @property
    def parser(self):
        parser = self._parser(add_help=True)
        grp = parser.add_argument_group(self._doc())
        self._add_argument(grp, 'lhs_criterion', 
                           choices=('center', 'c', 'maximin', 'm', 
                                    'centermaximin', 'cm', 'correlation', 'corr'))
        self._add_argument(grp, 'lhs_iterations', type=int)
        return parser


class Sample(Program, PriorConfig, LatinHyperCube):
    """Sample prior parameter distribution

    * method : Sampling method: Monte Carlo or Latin Hypercube Sampling 
    """
    def __init__(self, out=None, size=None, seed=None, method='lhs', **kwargs):
        self.out = out
        self.size = size
        self.seed = seed
        self.method = method
        self._super_init(self, **kwargs)

    @property
    def parser(self):
        parser = self._parser(description=self._doc())
        #parser.add_argument('-c', '--config-file', dest="config_file", help="configuration file")
        parser.add_argument('-o', '--out', help="output parameter file")

        parser.add_argument('-N', '--size',type=int, required=True, 
                          help="Sample size")
        parser.add_argument('--seed', type=int, 
                          help="random seed, for reproducible results (default to None)")
        self._add_argument(parser, 'method', choices=['montecarlo','lhs'])
        return parser

    def main(self):
        o = self
        prior = self.getprior()
        xparams = prior.sample(o.size, seed=o.seed, 
                               method=o.method,
                               criterion=o.lhs_criterion,
                               iterations=o.lhs_iterations)

        return return_params(xparams, self.out)

sample = Sample()

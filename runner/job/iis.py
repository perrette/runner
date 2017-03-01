"""Iterative Importance Sampling Calibration
"""
import argparse
from runner.param import Param, MultiParam
from runner.model import Model
from runner.iis import IISExp
from runner.job.setup import interface
from runner.job.tools import Job

parser = argparse.ArgumentParser(parents=[interface.parser], description=__doc__)
parser.add_argument('-o', '--out', default='.')
parser.add_argument('--niter', default=10, help="number of iterations", type=int)
parser.add_argument('-N', '--size', help="New sample size (default: same size as before)", type=int)
parser.add_argument('--seed', type=int, help="random seed, for reproducible results (default to None)")
x = parser.add_mutually_exclusive_group()
x.add_argument('--restart', action='store_true')
x.add_argument('--params-file', help="initial param file (sampled from prior if not provided)")
parser.add_argument('-f', '--force', action='store_true')
grp = parser.add_argument_group('iis')
grp.add_argument('--epsilon', help="", type=float)

grp = parser.add_argument_group('distributions')
grp.add_argument('-p', '--prior',
                 type=Param.parse,
                 default=[],
                 help="prior distribution of model parameters",
                 metavar="NAME=DIST",
                 nargs='+')

grp.add_argument('-l', '--likelihood',
                 type=Param.parse,
                 help="likelihood of model outputs",
                 metavar="NAME=DIST",
                 default=[],
                 nargs='+')


def run(o):

    initdir = os.path.join(o.out, 'runs')
    if o.restart:
        model = Model.read(o.out)
    else:
        model = Model(interface.get(o), o.prior, o.likelihood)
        model.write(o.out, o.force)

    iis = IISExp(model, initdir, epsilon=self.epsilon, seed=o.seed, size=o.size)

    if o.params_file:
        xparam = XParams.read(o.params_file)
        pfile = iis.path("params.txt")
        if os.path.exists(pfile):
            parser.error('param file already exists, use --restart')
        xparam.write(pfile)
        o.restart = True

    if o.restart:
        iis.restart(o.niter)
    else:
        iis.start(o.niter)


main = Job(parser, run)
main.register('iis', help='tuning: iterative importance samplping')

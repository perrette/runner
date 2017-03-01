#!/usr/bin/env python2.7
"""Analyze run results
"""
import argparse
from runner.param import ScipyParam, MultiParam
from runner.xrun import XRun
from runner.job.tools import Job
from runner.job.run import EXPDIR


analyze = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
analyze.add_argument('expdir', default=EXPDIR, 
                               help='experiment directory to analyze')
analyze.add_argument('--out', default=None,
                               help='experiment directory to write the diagnostics to (by default same as expdir)')

grp =analyze.add_argument_group("model output", description='')

grp.add_argument("-v", "--output-variables", nargs='+', default=[],
                 help='list of output variables to include in output.txt, \
                 does not necessarily enter in the likelihood')
grp.add_argument('--stats', action='store_true', help='add statistics on model output')

grp = analyze.add_argument_group(
    "likelihood", 
    description='likelihood is provided a list of distributions (same convention as job sample)')

grp.add_argument('-l', '--likelihood',
                 type=ScipyParam.parse,
                 help='NAME=SPEC where SPEC define a distribution: N?MEAN,STD or U?MIN,MAX or TYPE?ARG1[,ARG2 ...] \
        where TYPE is any scipy.stats distribution with *shp, loc, scale parameters.',
                 metavar="NAME=DIST",
                 default = [],
                 nargs='+')

grp.add_argument('-J', '--cost', nargs='+', default=[], help='output variables that shall be treated as the result of an objective (or cost) function, this is equivalent to have the likelihood N?0,1')


def analyze_post(o):

    # load namespace saved along with run command
    xrun = XRun.load(o.expdir)
    xrun.model.likelihood = MultiParam(o.likelihood + [Param.parse(name+"=N?0,1") for name in o.cost])
    xrun.analyze(o.output_variables, anadir=o.out)


analyze = Job(analyze, analyze_post)
analyze.register('analyze', help="analyze ensemble (output + loglik + stats) for resampling")

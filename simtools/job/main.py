#!/usr/bin/env python
"""Jobs for numerical experiments
"""
#import simtools.model.params as mp
import argparse
import warnings
import inspect
import copy
import logging
import json
import subprocess
import tempfile
import numpy as np
from simtools import __version__
from simtools.model import Model, Param
from simtools.prior import Prior, GenericParam, DiscreteParam
import simtools.sampling.resampling as xp
from simtools.xparams import XParams, Resampler
from simtools import register
import simtools.ext.namelist   # extensions
from simtools.job.model import model, getmodel

# prepare job
# ===========
register_job = register.register_job


# generate params.txt (XParams)
# =============================
def _return_params(xparams, out):
    "Return new ensemble parameters"
    if out:
        with open(out, "w") as f:
            f.write(str(xparams))
    else:
        print(str(xparams))

# product
# -------
product = argparse.ArgumentParser(add_help=False,
                                  description="Factorial combination of parameter values")
product.add_argument('factors',
                 type=DiscreteParam.parse,
                 metavar="NAME=VAL1[,VAL2 ...]",
                 nargs='+')
product.add_argument('-o','--out', help="output parameter file")


def product_post(o):
    xparams = Prior(o.factors).product()
    return _return_params(xparams, o.out)

register_job('product', product, product_post,
                 help='generate ensemble from all parameter combinations')


# sample
# ------
prior = argparse.ArgumentParser(add_help=False)
grp = prior.add_argument_group("prior distribution of model parameters")
grp.add_argument('dist',
                 type=GenericParam.parse,
                 help=GenericParam.parse.__doc__,
                 metavar="NAME=DIST",
                 nargs='+')

lhs = argparse.ArgumentParser(add_help=False)
grp = lhs.add_argument_group("Latin hypercube sampling")
grp.add_argument('--lhs-criterion', 
                   choices=('center', 'c', 'maximin', 'm', 
                            'centermaximin', 'cm', 'correlation', 'corr'), 
                 help='randomized by default')
grp.add_argument('--lhs_iterations', type=int)


sample = argparse.ArgumentParser(description="Sample prior parameter distribution", 
                                 add_help=False, parents=[prior, lhs])
sample.add_argument('-o', '--out', help="output parameter file")

sample.add_argument('-N', '--size',type=int, required=True, 
                  help="Sample size")
sample.add_argument('--seed', type=int, 
                  help="random seed, for reproducible results (default to None)")
sample.add_argument('--method', choices=['montecarlo','lhs'], default='lhs', 
                    help="sampling method (default=%(default)s)")

def sample_post(o):
    prior = Prior(o.dist)
    xparams = prior.sample(o.size, seed=o.seed, 
                           method=o.method,
                           criterion=o.lhs_criterion,
                           iterations=o.lhs_iterations)
    return _return_params(xparams, o.out)

register_job('sample', sample, sample_post,
                 help='generate ensemble by sampling prior distributions')


# resample
# --------
resample = argparse.ArgumentParser(add_help=False, description=xp.__doc__)
resample.add_argument("params_file", 
                    help="ensemble parameter flle to resample")

#grp = resample.add_argument_group('weights')
resample.add_argument('--weights-file', '-w', required=True, 
                   help='typically the likelihood from a bayesian analysis, i.e. exp(-((model - obs)**2/(2*variance), to be multiplied when several observations are used')
resample.add_argument('--log', action='store_true', 
                   help='set if weights are provided as log-likelihood (no exponential)')

grp = resample.add_argument_group('jittering')
grp.add_argument('--iis', action='store_true', 
                  help="IIS-type resampling with likelihood flattening + jitter")
grp.add_argument('--epsilon', type=float, 
                   help='Exponent to flatten the weights and derive jitter \
variance as a fraction of resampled parameter variance. \
    If not provided 0.05 is used as a starting value but adjusted if the \
effective ensemble size is not in the range specified by --neff-bounds.')

grp.add_argument('--neff-bounds', nargs=2, default=xp.NEFF_BOUNDS, type=int, 
                   help='Acceptable range for the effective ensemble size\
                   when --epsilon is not provided. Default to %(default)s.')

grp = resample.add_argument_group('sampling')
grp.add_argument('--method', choices=['residual', 'multinomial'], 
                   default=xp.RESAMPLING_METHOD, 
                   help='resampling method (default: %(default)s)')

grp.add_argument('-N', '--size', help="New sample size (default: same size as before)", type=int)
grp.add_argument('--seed', type=int, help="random seed, for reproducible results (default to None)")

grp = resample.add_argument_group('output')
grp.add_argument('-o', '--out', help="output parameter file (print to scree otherwise)")


def _getweights(o):
    w = np.loadtxt(o.weights_file)
    if o.log:
        w = np.exp(o.log)
    return w


def resample_post(o):
    weights = _getweights(o)
    xpin = XParams.read(o.params_file)
    xparams = xpin.resample(weights, size=o.size, seed=o.seed,
                            method=o.method,
                            iis=o.iis, epsilon=o.epsilon, 
                            neff_bounds=o.neff_bounds, 
                            )
    return _return_params(xparams, o)


register_job('resample', resample, resample_post,
                 help='resample parameters from previous simulation')


# TODO : implement 1 check or tool function that returns a number of things, such as neff
## check
## -----
#def neff(argv=None):
#    """Check effective ensemble size
#    """
#    parser = CustomParser(description=neff.__doc__, parents=[], 
#                          formatter_class=argparse.RawDescriptionHelpFormatter)
#    parser.add_argument('--weights-file', '-w', required=True, 
#                       help='typically the likelihood from a bayesian analysis, i.e. exp(-((model - obs)**2/(2*variance), to be multiplied when several observations are used')
#    parser.add_argument('--log', action='store_true', 
#                       help='set if weights are provided as log-likelihood (no exponential)')
#    parser.add_argument('--epsilon', type=float, default=1, 
#                      help='likelihood flattening, see resample sub-command')
#
#    args = parser.parse_args()
#    args.weights = getweights(args.weights_file, args.log)
#
#    print( Resampler(args.weights**args.epsilon).neff() )
#
#    #job.add_command("neff", neff, 
#    #                help='(resample helper) calculate effective ensemble size')



# run
# ---

def parse_slurm_array_indices(a):
    indices = []
    for i in a.split(","):
        if '-' in i:
            if ':' in i:
                i, step = i.split(':')
                step = int(step)
            else:
                step = 1
            start, stop = i.split('-')
            start = int(start)
            stop = int(stop) + 1  # last index is ignored in python
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(i))
    return indices

def _typechecker(type):
    def check(string):
        try:
            type(string) # just a check
        except Exception as error:
            print('ERROR:', error.message)
            raise
        return string

# SLURM high-performance computer
slurm = argparse.ArgumentParser(add_help=False)
grp = slurm.add_argument_group('slurm', 
                            description="These options only apply with --submit")
grp.add_argument('--qos', help='queue')
grp.add_argument('--job-name')
grp.add_argument('--account')
grp.add_argument('--walltime')

# 
submit = argparse.ArgumentParser(add_help=False)
grp = submit.add_argument_group("simulation mode (submit, background...)")
#grp.add_argument('--batch-script', help='')
#x = grp.add_mutually_exclusive_group()
grp.add_argument('-s', '--submit', action='store_true', help='submit job to slurm')
grp.add_argument('-t', '--test', action='store_true', 
               help='test mode: print to screen instead of log, run sequentially')
grp.add_argument('-w','--wait', action='store_true', help='wait for job to end')
grp.add_argument('-b', '--array', action='store_true', 
                 help='submit using sbatch --array (faster!), EXPERIMENTAL)')
grp.add_argument('--dry-run', action='store_true', 
                 help='might write a few files, but do not run')
#x.add_argument('--background', 
#                 action='store_true', help='run in the background, do not wait for executation to end')

simu = argparse.ArgumentParser(add_help=False)
grp = simu.add_argument_group("simulation settings")
simu.add_argument('-o','--out-dir', default='out',
                  help='experiment directory \
                  (params.txt and logs/ will be created, and possibly individual model output directories (each as {rundir})')
simu.add_argument('-a','--auto-folder', action='store_true', 
                 help='{runtag} and {rundir} named according to parameter values instead of {runid}')

x = simu.add_mutually_exclusive_group()
x.add_argument('-p', '--params',
                 type=DiscreteParam.parse,
                 help=DiscreteParam.parse.__doc__,
                 metavar="NAME=SPEC",
                 nargs='*')
x.add_argument('-i','--params-file', help='ensemble parameters file')
simu.add_argument('-j','--id', type=_typechecker(parse_slurm_array_indices), dest='runid', 
                 metavar="I,J...,START-STOP:STEP,...",
                 help='select one or several ensemble members (0-based !), \
slurm sbatch --array syntax, e.g. `0,2,4` or `0-4:2` \
    or a combination of these, `0,2,4,5` <==> `0-4:2,5`')

#run = argparse.ArgumentParser(add_help=False, parents=[model, simu, submit, slurm],
run = argparse.ArgumentParser(add_help=False, parents=[model, simu, submit, slurm],
                              description='run model (single version or ensemble)')


# sub
_slurmarray = argparse.ArgumentParser(add_help=False, parents=[model, simu])
_slurmarray_defaults = {a.dest:a.default for a in _slurmarray._actions}  # default arguments


def _autodir(params):
    " create automatic directory based on list of Param instances"
    raise NotImplementedError()

def run_post(o):
    model = getmodel(o)  # default model

    if o.params_file:
        xparams = XParams.read(o.params_file)
    elif o.params:
        prior = Prior(o.params)
        xparams = prior.product() # only product allowed as direct input
        #update = {p.name:p.value for p in o.params}
    else:
        xparams = XParams(np.empty((0,0)), names=[])

    xrun = XRun(model, xparams)
    
    if o.id:
        indices = parse_slurm_array_indices(o.id)
    else:
        indices = np.arange(xparams.size)

    # test: run everything serially
    if o.test:
        for i in indices:
            rundir = '--auto' if self.auto_dir else None
            xrun.run(runid=i, expdir=o.out_dir, background=False, rundir=rundir, dry_run=o.dry_run)
        o.wait = False

    # array: create a parameterized "job" command [SLURM]
    elif o.array:
        # input via params file
        if not os.path.exists(o.out_dir):
            os.makedirs(o.out_dir)
        params_file = os.path.join(o.out_dir, 'params.txt')
        xrun.params.write(params_file)

        # prepare job command: runid and params passed by slurm
        cfg = o.__dict__.copy()
        del cfg["params"] # passed by file
        del cfg["params_file"] # see below
        del cfg["id"] # see below

        # write command based on namespace state
        file = tempfile.mktemp(dir=o.out_dir, prefix='job.run-array.', suffix='.json')
        write_config(cfg, file, defaults=_slurmarray_defaults, diff=True, name="run")
        template = "{job} -c {config_file} run --id $SLURM_ARRAY_TASK_ID --params-file {params_file}"
        command = template.format(job="job", config_file=file, params_file=params_file) 
        #FIXME: job may be a full path `/path/to/job run` or called via `[/path/to/]python job run`
        #TODO: maybe figure out something?

        # slurm-related options are passed directyl
        slurm_opt = {a.dest:a.default for a in slurm._actions if a.default is not None}
        slurm_opt["array"] = o.id or "{}-{}".format(0, xparams.size-1)

        if o.dry_run:
            print(slurm_opt)
            print(command)
            return

        p = submit_job(command, **slurm_opt)

    # the default
    else:
        p = xrun.batch(self, indices=indices, submit=o.submit, 
                   expdir=o.out_dir, autodir=o.auto_dir) #, output=o.log_out, error=o.log_err)

    if o.wait:
        p.wait()

    return


register_job('run', run, run_post, help='ensemble run')

#obs = argparse.ArgumentParser(add_help=False, description="observational constraints")
#obs.add_argument('--likelihood', '-l', dest='constraints',
#                 type=typechecker(GenericParam.parse),
#                 help=GenericParam.parse.__doc__,
#                 metavar="NAME=SPEC",
#                 nargs='*')


# job config I/O
# ==============
def _parser_defaults(parser):
    " parser default values "
    return {a.dest: a.default for a in parser._actions}


def _modified(kw, defaults):
    """return key-words that are different from default parser values
    """
    return {k:kw[k] for k in kw if k in defaults and kw[k] != defaults[k]}

def _filter(kw, after, diff=False, include_none=True):
    if diff:
        filtered = _modified(kw, after)
    else:
        filtered = {k:kw[k] for k in kw if k in after}
    if not include_none:
        filtered = {k:filtered[k] for k in filtered if filtered[k] is not None}
    return

def json_config(cfg, defaults=None, diff=False, name=None):
    import datetime
    js = {
        'defaults': _filter(cfg, defaults, diff) if defaults is not None else cfg,
        'version':__version__,
        'date':str(datetime.date.today()),
        'name':name,  # just as metadata
    }
    return json.dumps(js, indent=2, sort_keys=True, default=lambda x: str(x))

def write_config(cfg, file, defaults=None, diff=False, name=None):
    string = json_config(cfg, defaults, diff, name)
    with open(file, 'w') as f:
        f.write(string)



# pull main job together
# ======================
# savable config
#global_defaults = {}
#global_defaults.update(_parser_defaults(slurm))
#global_defaults.update(_parser_defaults(model))


# prepare parser
job = argparse.ArgumentParser(parents=[], description=__doc__, 
                              formatter_class=argparse.RawTextHelpFormatter)
job.add_argument('-c','--config-file', 
                    help='load defaults from configuration file')
x = job.add_mutually_exclusive_group()
x.add_argument('-s','--saveas', action="store_true", 
               help='save selected defaults to config file and exit')
x.add_argument('-u', '--update-config', action="store_true", 
                    help='-uc FILE is an alias for -c FILE -s FILE')
job.add_argument('--show', action="store_true", help='show config and exit')


def main(argv=None):

    # add subcommands
    subp = job.add_subparsers(dest='cmd')
    postprocs = {}
    parsers = {}

    for j in register.jobs:
        subp.add_parser(j.name, parents=[j.parser], help=j.help)
        parsers[j.name] = j.parser
        postprocs[j.name] = j.postproc


    # parse arguments and select sub-parser
    o = job.parse_args(argv)
    parser = parsers[o.cmd]
    func = postprocs[o.cmd]

    # read config file?
    if o.config_file:

        js = json.load(open(o.config_file))
        if js["name"] != o.cmd:
            warnings.warn("config file created from another command")

        parser.set_defaults(**js["defaults"])

        update, unknown = parser.parse_known_args(argv)  
        o.__dict__.update(update.__dict__)

    if o.update_config:
        o.saveas = o.config_file

    # save to file?
    if o.saveas or o.show:
        #saveable = _filter(o.__dict__, global_defaults, diff=False, include_none=False)
        saveable = _filter(o.__dict__, _parser_defaults(parser), diff=False, include_none=False)
        string = json_config(saveable, name=o.cmd)
        if o.saveas:
            with open(o.saveas, 'w') as f:
                f.write(string)
        if o.show:
            print(string)
        return

    return func(o)

if __name__ == '__main__':
    main()

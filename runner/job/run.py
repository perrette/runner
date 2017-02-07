"""Model ensemble run
"""
import argparse
import tempfile
import numpy as np
from runner.prior import Prior, DiscreteParam
#from runner.xparams import XParams
from runner.xrun import XParams, XRun, XPARAM
from runner.submit import submit_job
from runner import register
from runner.job.model import model_parser as model, modelwrapper, getmodel, modelconfig
import runner.job.stats  # register !
from runner.job.config import write_config, json_config, filtervars
import os


EXPCONFIG = 'experiment.json'
EXPDIR = 'out'


# prepare job
# ===========
register_job = register.register_job
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
grp.add_argument('--shell', action='store_true', 
               help='print output to terminal instead of log file, run sequentially, mostly useful for testing/debugging')
grp.add_argument('--echo', action='store_true', 
                 help='display commands instead of running them (but does setup output directory). Alias for --shell --force echo [model args ...]')
grp.add_argument('-w','--wait', action='store_true', help='wait for job to end')
grp.add_argument('-b', '--array', action='store_true', 
                 help='submit using sbatch --array (faster!), EXPERIMENTAL)')
grp.add_argument('-f', '--force', action='store_true', 
                 help='perform run even in an existing directory')

folders = argparse.ArgumentParser(add_help=False)
grp = folders.add_argument_group("simulation settings")
grp.add_argument('-o','--out-dir', default=EXPDIR, dest='expdir',
                  help='experiment directory \
                  (params.txt and logs/ will be created, as well as individual model output directories')
grp.add_argument('-a','--auto-dir', action='store_true', 
                 help='run directory named according to parameter values instead of run `id`')

params_parser = argparse.ArgumentParser(add_help=False)
x = params_parser.add_mutually_exclusive_group()
x.add_argument('-p', '--params',
                 type=DiscreteParam.parse,
                 help=DiscreteParam.parse.__doc__,
                 metavar="NAME=SPEC",
                 nargs='*')
x.add_argument('-i','--params-file', help='ensemble parameters file')
params_parser.add_argument('-j','--id', type=_typechecker(parse_slurm_array_indices), dest='runid', 
                 metavar="I,J...,START-STOP:STEP,...",
                 help='select one or several ensemble members (0-based !), \
slurm sbatch --array syntax, e.g. `0,2,4` or `0-4:2` \
    or a combination of these, `0,2,4,5` <==> `0-4:2,5`')
params_parser.add_argument('--include-default', 
                  action='store_true', 
                  help='also run default model version (with no parameters)')

run = argparse.ArgumentParser(add_help=False, 
                              parents=[model, params_parser, folders, submit, slurm],
                              description=__doc__)


# keep group of params for later
experiment = argparse.ArgumentParser(add_help=False, parents=[modelconfig])
experiment.add_argument('-a','--auto-dir', action='store_true')

# ...only when --array is invoked
_slurmarray = argparse.ArgumentParser(add_help=False, parents=[model, folders])


def run_post(o):

    if o.echo:
        o.model = ['echo'] + o.model
        o.shell = True
        o.force = True

    model = getmodel(o)  # default model

    if o.params_file:
        xparams = XParams.read(o.params_file)
    elif o.params:
        prior = Prior(o.params)
        xparams = prior.product() # only product allowed as direct input
        #update = {p.name:p.value for p in o.params}
    else:
        xparams = XParams(np.empty((0,0)), names=[])
        o.include_default = True

    xrun = XRun(model, xparams, autodir=o.auto_dir)
    # create dir, write params.txt file, as well as experiment configuration
    try:
        xrun.setup(o.expdir, force=o.force)  
        pfile = os.path.join(o.expdir, XPARAM)
    except RuntimeError as error:
        print("ERROR :: "+error.message)
        print("Use -f/--force to bypass this check")
        run.exit(1)

    write_config(vars(o), os.path.join(o.expdir, EXPCONFIG), parser=experiment)

    if o.runid:
        indices = parse_slurm_array_indices(o.runid)
    else:
        indices = np.arange(xparams.size)

    if o.include_default:
        indices = list(indices) + [None]

    slurm_opt = filtervars(o, slurm, include_none=False)

    # test: run everything serially
    if o.shell:
        for i in indices:
            model = xrun.get_model(i)
            rundir = xrun.get_rundir(i, o.expdir)
            model.run(rundir, background=False)

    # array: create a parameterized "job" command [SLURM]
    elif o.array:
        # prepare job command: runid and params passed by slurm
        #base = tempfile.mktemp(dir=o.expdir, prefix='job.run-array.')
        base = os.path.join(o.expdir, 'job.run.array')
        file = base + '.json'
        script = base + '.sh'
        output = base + '.out'
        error = base + '.err'
        write_config(vars(o), file, parser=_slurmarray)
        template = "{job} -c {config_file} run --id $SLURM_ARRAY_TASK_ID --params-file {params_file} --force"
        command = template.format(job="job", config_file=file, params_file=pfile) 
        slurm_opt["array"] = o.runid or "{}-{}".format(0, xparams.size-1)
        p = submit_job(command, jobfile=script, output=output, error=error, **slurm_opt)

    # the default
    else:
        assert not slurm_opt.pop('array', False), 'missed if then else --array????'
        if o.submit:
            p = xrun.submit(indices=indices, expdir=o.expdir, **slurm_opt)
        else:
            p = xrun.run(indices=indices, expdir=o.expdir)

    if o.wait:
        p.wait()

    return


register_job('run', run, run_post, help='run model (single version or ensemble)')

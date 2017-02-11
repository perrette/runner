"""Run model ensemble

The canonical form of `job run` is:

    job run [OPTIONS] -- EXECUTABLE [OPTIONS]

where `EXECUTABLE` is your model executable or a command, followed by its
arguments. Note the `--` that separates `job run` arguments `OPTIONS` from the
executable.  When there is no ambiguity in the command-line arguments (as seen
by python's argparse) it may be dropped. `job run` options determine in which
manner to run the model, which parameter values to vary (the ensemble), and how
to communicate these parameter values to the model.
"""
examples="""
Examples
--------

    job run -p a=2,3,4 b=0,1 -o out --shell -- echo --a {a} --b {b} --out {}

    --a 2 --b 0 --out out/0
    --a 2 --b 1 --out out/1
    --a 3 --b 0 --out out/2
    --a 3 --b 1 --out out/3
    --a 4 --b 0 --out out/4
    --a 4 --b 1 --out out/5

The command above runs an ensemble of 6 model versions, by calling `echo --a {a}
--b {b} --out {}`  where `{a}`, `{b}` and `{}` are formatted using runtime with
parameter and run directory values, as displayed in the output above. Parameters can also be provided as a file:

    job run -p a=2,3,4 b=0,1 -o out --file-name "params.txt" --file-type "linesep" --line-sep " " --shell cat {}/params.txt

    a 2
    b 0
    a 2
    b 1
    a 3
    b 0
    a 3
    b 1
    a 4
    b 0
    a 4
    b 1

Where UNIX `cat` command displays file content into the terminal. File types
that involve grouping, such as namelist, require a group prefix with a `.`
separator in the parameter name:

    job run -p g1.a=0,1 g2.b=2. -o out --file-name "params.txt" --file-type "namelist" --shell  cat {}/params.txt

    &g1
     a               = 0          
    /
    &g2
     b               = 2.0        
    /
    &g1
     a               = 1          
    /
    &g2
     b               = 2.0        
    /
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
            print('ERROR:', str(error))
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

run = argparse.ArgumentParser(parents=[model, params_parser, folders, submit, slurm], epilog=examples, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)


# keep group of params for later
experiment = argparse.ArgumentParser(add_help=False, parents=[modelconfig, modelwrapper])
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
        print("ERROR :: "+str(error))
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

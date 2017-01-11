#!/usr/bin/env python
"""Run model based on parameters defined in config.json

Note that they do not need to be the same as the original model, 
for instance parameters may be transform (e.g. inverse) to obtain
a more linear behaviour w.r.t the output.
"""
import os
import json
from collections import OrderedDict as odict
import numpy as np
import subprocess

def read_ensemble_params(pfile):
    pnames = open(pfile).readline().split()
    pvalues = np.loadtxt(pfile, skiprows=1)  
    return pnames, pvalues


def run_background(cmd=(), ini_dir='.', logfile='out.out'):
    " execute in the background "
    cmdstr = " ".join(cmd)
    print("Running job in background: %s" % (cmd[0]))
    print("...initial directory : %s" % (ini_dir))
    print("...log file : %s" % (logfile))
    #print "Storing output in: %s" % (out_dir)
    cmdstr = "%s > '%s' 2>&1 &" % (cmdstr, logfile)
    if ini_dir != os.path.curdir:
        cmdstr = "cd '%s' && " % (ini_dir) + cmdstr  # go to initial directory prior to execution

    code = os.system (cmdstr)
    return code

def run_foreground(cmd=(), ini_dir='.', logfile=None):
    " execute in terminal, with blocking behaviour "
    cmdstr = " ".join(cmd) #if not isinstance(cmd, basestring) else cmd

    # execute from directory...
    if ini_dir != os.path.curdir:
        cmdstr = "cd '%s' && " % (ini_dir) + cmdstr

    if logfile is not None:
        cmdstr = cmdstr + " 2>&1 | tee "+logfile

    code = os.system (cmdstr)
    return code


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


def make_jobfile_slurm(command, queue, jobname, account, output, error, time):
    return """#!/bin/bash

#SBATCH --qos={queue}
#SBATCH --job-name={jobname}
#SBATCH --account={account}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --time={time}

echo
echo SLURM JOB
echo ---------
echo "SLURM_JOBID $SLURM_JOBID"
echo "SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID"
cmd="{command}"
echo $cmd
eval $cmd""".format(**locals())


def wait_for_jobid(jobid, freq=1):
    """wait until job completion
    """
    import time
    cmd="squeue --job {jobid} | grep -q {jobid}".format(jobid=jobid)
    while True:
        if os.system(cmd) == 0:
            # job still in the queue, wait
            time.sleep(freq)
        else:
            # not found return
            return


class HPC(object):
    pass


class Slurm(HPC):
    def __init__(self, **kwargs):
        """
        kwargs : keyword arguments, e.g. job_name="bla"
        """
        self.opt = kwargs

    def header_commands(self, **kwargs):
        "commands that would end up in the script header"
        cmd = []
        opt = self.opt.copy()
        opt.update(kwargs)
        for k in opt:
            nm = '--'+k.replace('_','-')
            val = opt[k]
            cmd.append(nm + " " + val)
        return cmd


    def header(self, **kwargs):
        header = ["#!/bin/bash"]
        for c in self.header_command(**kwargs):
            header.append("#SBATCH "+c)
        return "\n".join(header)


    def jobscript(self, cmd, array=None, verbose=True, **kwargs):
        lines = [self.header(**kwargs)]

        if verbose:
            lines = ["echo"]
            lines = ["echo SLURM JOB"]
            lines = ["echo ---------"]
            lines = ["SLURM_JOBID $SLURM_JOBID"]
            lines = ["SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID"]
            lines = ["SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID"]

        lines.append(" ".join(cmd))
        #lines.append("cmd="+" ".join(cmd))
        #lines.append("echo $cmd")
        #lines.append("eval $cmd")

        if verbose:
            lines.append("echo SLURM JOB DONE")
        return "\n".join(lines)
        

    def submit_job(self, cmd, array=None, jobfile=None, wait=False, **kwargs):

        if jobfile is None:
            import tempfile
            jobfile = tempfile.mktemp(prefix='jobfile.', suffix='.sh')
            print("Creating temporary jobfile:",jobfile)

        jobtxt = self.jobscript(cmd, array, **kwargs)

        # command to submit the job
        if array is not None:
            arr = [ "--array", array]
        else:
            arr = []
        batchcmd = ["sbatch"] + arr + [jobfile]

        # indicate submit command as a comment in jobfile
        jobtxt += '\n#NOTE::to submit:: '+" ".join(batchcmd)

        # write down and submit
        with open(jobfile, "w") as f:
            f.write(jobtxt)

        output = subprocess.check_output(batchcmd)
        jobid = output.split()[-1]

        if wait:
            wait_for_jobid(jobid)

        return jobid


def run_command(cmd, rundir, dry_run=False, background=False, submit=False, **kwargs):
    """Execute a command
    """
    print(" ".join(cmd))
    
    if dry_run:
        return

    if not os.path.exists(rundir):
        os.makedirs(rundir)

    logfile = os.path.join(rundir, "run.log")
    jobfile = os.path.join(rundir, "run.cmd")

    if background:
        ret = run_background(cmd, logfile=logfile)
        if kwargs.pop("wait", False):
            raise NotImplementedError("background + wait")

    elif submit:
        slurm = Slurm()
        kwargs["output"] = kwargs.pop("output", logfile)
        kwargs["error"] = kwargs.pop("error", logfile)
        kwargs["jobfile"] = kwargs.pop("jobfile",jobfile)
        ret = slurm.submit_job(cmd, **kwargs)

    else:
        ret = run_foreground(cmd, logfile=logfile)

    return ret

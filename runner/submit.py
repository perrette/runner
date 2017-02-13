"""Submit job to High Performance Computer
"""
import os
import subprocess
import tempfile
import six

MANAGER = "slurm"

# To submit the model
# ===================
class JobScript(object):

    interpreter = "#!/bin/bash"

    def __init__(self, commands, env=None, **opt):

        if type(commands) in six.string_types:
            commands = commands.splitlines()
        
        self.opt = opt
        self.lines = []

        # export environment variables
        env = env or {}
        for k in sorted(env.keys()):
            self.lines.append("export "+k+'='+env[k])

        # add commands
        for cmd in commands:
            assert type(cmd) in six.string_types, "commands must be strings"
            self.lines.append(cmd)

    @property
    def header(self):
        return ""

    @property
    def body(self):
        return "\n".join(self.lines)

    @property
    def script(self):
        return "\n".join([self.interpreter,"", self.header, "", self.body])

    def submit(self, jobfile, **kwargs):
        opt = self.opt.copy()
        opt.update(kwargs)
        return subprocess.Popen(["bash", jobfile], **opt)


class Slurm(JobScript):
    """specific for Slurm manager
    """
    def make_arg(self, name, value):
        if name.startswith('-'):
            return name + " " + str(value)
        else:
            return '--{} {}'.format(name.replace('_','-'), value)

    @property
    def header(self):
        lines = []
        for k in self.opt:
            lines.append("#SBATCH "+self.make_arg(k, self.opt[k]))
        return "\n".join(lines)

        
    def submit(self, jobfile, **kwargs):
        """Submit job and return jobid
        """
        args = [self.make_arg(k, kwargs[k]) for k in kwargs]
        batchcmd = ["sbatch"] + args + [jobfile]
        output = subprocess.check_output(batchcmd)
        jobid = output.split()[-1]
        return SlurmProcess(jobid)


class SlurmProcess(object):
    def __init__(self, jobid):
        self.jobid = jobid
        self.returncode = None

    def running(self):
        cmd="sacct --job {jobid} | grep -q RUNNING".format(jobid=self.jobid)
        return subprocess.call(cmd, shell=True) == 0

    def completed(self):
        cmd="sacct --job {jobid} | grep -q COMPLETED".format(jobid=self.jobid)
        return subprocess.call(cmd, shell=True) == 0

    def failed(self):
        cmd="sacct --job {jobid} | grep -q FAILED".format(jobid=self.jobid)
        return subprocess.call(cmd, shell=True) == 0

    def wait(self, freq=1):
        import time
        while self.running():
            time.sleep(freq)
        self.returncode = 0 if self.completed() else 1
        return self.returncode

    def kill(self):
        return subprocess.call("scancel {jobid}".format(jobid=self.jobid))



def submit_job(commands, manager=MANAGER, jobfile=None, 
               output=None, error=None, workdir=None, **kwargs):
    """Write a series of command to file and execute them

    commands : [str] or str
        list of (string) commands to be written to a file
    manager : str, optional
        job manager ("slurm" or None) so far
    jobfile : job script to be written
    output, error : log files (str)
    **kwargs : other, manager-specific arguments
    """
    opt = {}  # to be passed on submit

    if manager is None:
        # make it behave more like SLURM
        if output: kwargs["stdout"] = output
        if error: kwargs["stderr"] = error
        if workdir: 
            kwargs["cwd"] = workdir
        job = JobScript(commands, **kwargs)

    elif manager == "slurm":
        if workdir: 
            opt["workdir"] = workdir
        if output: kwargs["output"] = output
        if error: kwargs["error"] = error
        job = Slurm(commands, **kwargs)

    else:
        raise ValueError("unknown job manager:"+manager)

    if jobfile is None:
        jobfile = tempfile.mktemp(prefix='jobfile', suffix='.sh')

    # write down and submit
    with open(jobfile, "w") as f:
        f.write(job.script)

    p = job.submit(jobfile, **opt)

    return p

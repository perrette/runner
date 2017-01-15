"""Submit job to High Performance Computer
"""
import os
import subprocess
import tempfile


# To submit the model
# ===================
class JobScript(object):

    interpreter = "#!/bin/bash"

    def __init__(self, commands, env=None, **opt):
        
        self.opt = opt
        self.lines = []

        # export environment variables
        env = env or []
        for k in sorted(env.keys()):
            self.lines.append("export "+k+'='+env[k])

        # add commands
        for cmd in commands:
            assert isinstance(cmd, basestring), "commands must be strings"
            self.lines.append(cmd)

    @property
    def header(self):
        return ""

    @property
    def body(self):
        return "\n".join([self.lines])

    @property
    def script(self):
        return "\n".join([self.interpreter,"", self.header, "", self.body])

    def submit(self, jobfile, **kwargs):
        opt = self.opt.copy()
        opt.update(kwargs)
        return subprocess.Popen([jobfile], **opt)


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
            lines.append("#SBATCH "+self.make_arg(k, opt[k]))
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



def submit_job(commands, jobfile=None, manager=None, **kwargs):
    """Write a series of command to file and execute them

    commands : [str]
        list of (string) commands to be written to a file

    """
    if manager is None:
        job = JobScript(commands, **kwargs)

    elif manager == "slurm":
        job = Slurm(commands, **kwargs)

    else:
        raise ValueError("unknown job manager:"+manager)

    if jobfile is None:
        jobfile = tempfile.mktemp(prefix='jobfile', suffix='.sh')

    # write down and submit
    with open(jobfile, "w") as f:
        f.write(job.script)

    p = job.submit(jobfile)

    return p

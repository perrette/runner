from __future__ import print_function, absolute_import
import subprocess
import os
import logging
import sys
import json
import datetime
from collections import OrderedDict as odict, namedtuple
import six
from argparse import Namespace
from runner import __version__
from runner.param import Param, MultiParam
from runner.tools import parse_val
from runner.submit import submit_job
#from runner.model.generic import get_or_make_filetype

# default values
ENV_OUT = "RUNDIR"


ParamIO = namedtuple("ParamIO", ["name","value"])


class ParamsFile(object):
    """Parent class for the parameters
    """
    def dumps(self, params):
        raise NotImplementedError()

    def loads(self, string):
        raise NotImplementedError()

    def dump(self, params, f):
        f.write(self.dumps(params))

    def load(self, f):
        return self.loads(f.read())


class ModelInterface(object):
    def __init__(self, args=None, 
                 filetype=None, filename=None, 
                 arg_out_prefix=None, arg_param_prefix=None, 
                 env_out=ENV_OUT, env_prefix=None,
                 work_dir=None, 
                 filetype_output=None, filename_output=None,
                 ):
        """
        * args : [str] or str
            Executable and command arguments. This command may contain the `{}` tag for model run
            directory, and any `{NAME}` for parameter names. Alternatively these
            might be set with `arg_out_prefix` and `arg_param_prefix` options.
        * filetype : ParamsFile instance or anything with `dump` method, optional
        * filename : relative path to rundir, optional
            filename for parameter passing to model (also needs filetype)
        * arg_out_prefix : str, optional
            prefix for command-line passing of output dir (e.g. "" or "--out ")
        * arg_param_prefix : str, optional
            prefix for command-line passing of one parameter, e.g. "--{}"
        * env_out : str, optional
            environment variable name for output directory
        * env_prefix : str, optional
            environment passing of parameters, e.g. "RUNNER_" to be completed
            with parameter name or RUNDIR for model output directory.
        * work_dir: str, optional
            directory to start the model from (work directory)
            by default from the current directory
        * filetype_output : ParamsFile instance or anything with `load` method, optional
        * filename_output : relative path to rundir, optional
            filename for output variable (also needs filetype_output)
        """
        if type(args) in six.string_types:
            args = args.split()
        self.args = args or []
        self.filetype = filetype
        self.filename = filename
        self.filetype_output = filetype_output
        self.filename_output = filename_output
        self.arg_out_prefix = arg_out_prefix
        self.arg_param_prefix = arg_param_prefix
        self.env_prefix = env_prefix
        self.env_out = env_out
        self.work_dir = work_dir or os.getcwd() 

        # check !
        if filename:
            if filetype is None: 
                raise ValueError("need to provide FileType with filename")
            if not hasattr(filetype, "dumps"):
                raise TypeError("invalid filetype: no `dumps` method: "+repr(filetype))


    def _command_out(self, rundir):
        if self.arg_out_prefix is None:
            return []
        return (self.arg_out_prefix + rundir).split() 

    def _command_param(self, name, value):
        if self.arg_param_prefix is None:
            return []
        prefix = self.arg_param_prefix.format(name, value)
        return (prefix + str(value)).split()

    def _format_args(self, rundir, **params_kw):
        """two-pass formatting: first rundir and params with `{}` and `{NAME}`
        then `{{rundir}}`
        """
        return [arg.format(rundir, **params_kw).format(rundir=rundir) 
                for arg in self.args[1:]]

    def command(self, rundir, params_kw):
        if not self.args:
            raise ValueError("model requires an executable")

        exe = self.args[0]

        if os.path.isfile(exe):
            if not os.access(exe, os.X_OK):
                raise ValueError("model executable is not : check permissions")

        args = [exe] 
        args += self._command_out(rundir)
        args += self._format_args(rundir, **params_kw)

        # prepare modified command-line arguments with appropriate format
        for name, value in params_kw.items():
            args += self._command_param(name, value)

        return args

    def environ(self, rundir, params_kw, env=None):
        """define environment variables to pass to model
        """
        if self.env_prefix is None:
            return None

        # prepare variables to pass to environment
        context = {}
        if self.env_out is not None:
            context[self.env_out] = rundir 
        context.update(params_kw)

        # format them with appropriate prefix
        update = {self.env_prefix+k:str(context[k])
               for k in context if context[k] is not None}

        # update base environment
        env = env or {}
        env.update(update)

        return env

    def workdir(self, rundir):
        """directory from which command is called, default to current (caller) directory (NOT rundir)
        """
        return self.work_dir.format(rundir)


    def runfile(self, rundir):
        return os.path.join(rundir, "runner.json")

    def _write(self, rundir, runinfo, update=False):

        if not os.path.isdir(rundir):
            runfile = rundir
        else:
            runfile = self.runfile(rundir)

        if update:
            updateinfo = runinfo
            runinfo = json.load(open(runfile))
            runinfo.update(updateinfo)

        # add metadata
        runinfo['time'] = str(datetime.datetime.now())
        runinfo['version'] = __version__
        runinfo['rundir'] = rundir

        with open(runfile, 'w') as f:
            json.dump(runinfo, f, 
                      indent=2, 
                      default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    def setup(self, rundir, params_kw):
        """Write param file to run directory (assumed already created)
        can be subclassed by the user
        """
        # write param file to rundir
        if self.filename:
            assert self.filetype
            #TODO: rename filename --> file_in OR file_param
            filepath = os.path.join(rundir, self.filename)
            #TODO: filetype: have model params as a dictionary
            paramlist = [ParamIO(k, params_kw[k]) for k in params_kw]
            self.filetype.dump(paramlist, open(filepath, 'w'))


    def run(self, rundir, params_kw, background=True, submit=False, **kwargs):
        """Run the model (background subprocess, submit to slurm...)

        - create directory if not existing
        - setup() : write param file if needed
        - call subprocess or submit to SLURM
        - postprocess() : read output
        - write runner.json
        """
        # create run directory
        if not os.path.exists(rundir):
            os.makedirs(rundir)

        args = self.command(rundir, params_kw)
        workdir = self.workdir(rundir)
        env = self.environ(rundir, params_kw)

        # also write parameters in a format runner understands, for the record
        info = odict()
        info['command'] = args
        info['workdir'] = workdir
        info['env'] = env
        info['params'] = params_kw
        info['status'] = 'running'
        self._write(rundir, info)

        self.setup(rundir, params_kw)

        output = kwargs.pop('output', os.path.join(rundir, 'log.out'))
        error = kwargs.pop('error', os.path.join(rundir, 'log.err'))

        if submit:
            jobfile = kwargs.pop('jobfile', os.path.join(rundir, 'submit.sh'))
            p = submit_job(" ".join(args), env=env, workdir=workdir, 
                              output=output, error=error, jobfile=jobfile, **kwargs)

        else:
            if background:
                stdout = open(output, 'w')
                stderr = open(error, 'w')
            else:
                stdout = None
                stderr = None

            try:
                if env is not None:
                    env2 = os.environ.copy()
                    env2.update(env)
                else:
                    env2=None
                p = subprocess.Popen(args, env=env2, cwd=workdir, 
                                     stdout=stdout, stderr=stderr)
            except OSError as error:
                raise OSError("FAILED TO EXECUTE: `"+" ".join(args)+"` FROM `"+workdir+"`")

        # wait for execution and postprocess
        ret = p.wait()

        if ret == 0:
            info['status'] = 'success'
            try:
                info['output'] = output = self.postprocess(rundir)
            except Exception as error:
                logging.warn("error during postprocessing: "+str(error))
            self._write(rundir, info)
        else:
            info['status'] = 'failed'
            self._write(rundir, info)
            raise RuntimeError('model run failed')

        return output


    def postprocess(self, rundir):
        """return model output as dictionary or None
        """
        if not self.filename_output:
            info = json.load(open(self.runfile(rundir)))
            return info.pop("output", {})

        assert self.filetype_output, "filetype_output is required"
        variables = self.filetype_output.load(open(os.path.join(rundir, self.filename_output)))
        return odict([(v.name,v.value) for v in variables])



class Model(object):
    """Bayesian model, where prior represents information about the parameters, 
    and posterior about output variables.
    """
    def __init__(self, interface=None, prior=None, likelihood=None):
        """
        * interface : ModelInterface instance
        * prior : [Param], optional
            list of model parameters distributions
        * likelihood : [Param], optional
            list of model output variables distributions (output)
        """
        self.interface = interface
        self.prior = MultiParam(prior or [])
        self.likelihood = MultiParam(likelihood or [])

    def __call__(self, rundir, params, output=None):
        """freeze model with rundir and params
        """
        # replace missing params with default values, get rid of params that are not defined
        params = self.prior(**params).as_dict()
        output = self.likelihood(**(output or {})).as_dict()
        return FrozenModel(self, rundir, params, output)


class FrozenModel(object):
    """'Frozen' model instance representing a model run, with fixed rundir, params and output variables
    """
    def __init__(self, model, rundir, params, output):
        """
        params : parameters as (possibly ordered) dict

        Example : FrozenModel(model, rundir, a=2, b=2, output=4)
        """
        self.model = model
        self.rundir = rundir
        self.params = params
        self.output = output
        self.status = None

    @property
    def prior(self):
        """prior parameter distribution
        """
        return self.model.prior(**self.params)

    @property
    def likelihood(self):
        """output variables' likelihood
        """
        return self.model.likelihood(**self.output)

    @property
    def posterior(self):
        """model posterior (params' prior * output posterior)
        """
        return self.prior + self.likelihood


    @property
    def runfile(self):
        return self.model.interface.runfile(self.rundir)

    def load(self, file=None):
        " load model output + params from output directory "
        cfg = json.load(open(file or self.runfile))
        self.params = cfg["params"]
        self.output = cfg.pop("output",{})
        self.status = cfg.pop("status", None)
        return self

    def save(self, file=None):
        " save / update model params + output "
        self.model.interface._write(self.rundir, {
            'output':self.output, 
            'params':self.params,
        }, update=True)


    def run(self, background=True, submit=False, **kwargs):
        """Run the model (background subprocess, submit to slurm...)
        """
        self.output = self.model.interface.run(self.rundir, self.params, submit=submit, background=background, **kwargs)
        self.status = "success"
        return self


    def postprocess(self):
        self.output = self.model.interface.postprocess(self.rundir)
        self.save()
        return self


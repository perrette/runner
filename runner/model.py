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


# Model instance
# ==============
class Model(object):
    def __init__(self, executable=None, args=None, params=None, state=None,
                 filetype=None, filename=None, 
                 arg_out_prefix=None, arg_param_prefix=None, 
                 env_out=ENV_OUT, env_prefix=None,
                 work_dir=None, 
                 filetype_output=None, filename_output=None,
                 ):
        """
        * executable : runscript
        * args : [str] or str, optional
            List of command arguments to pass to executable as general model
            configuration. This command may contain the `{}` tag for model run
            directory, and any `{NAME}` for parameter names. Alternatively these
            might be set with `arg_out_prefix` and `arg_param_prefix` options.
        * params : [Param], optional
            list of model parameters dist to be updated with modified params
        * state : [Param], optional
            list of model state variables dist (output)
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
        self.executable = executable
        if type(args) in six.string_types:
            args = args.split()
        self.args = args or []
        self.params = MultiParam(params or [])
        self.state = MultiParam(state or [])
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
                for arg in self.args]


    def command(self, rundir, params_kw):
        exe = self.executable
        if exe is None:
            raise ValueError("model requires an executable")
        elif os.path.isfile(exe):
            if not os.access(exe, os.X_OK):
                raise ValueError("model executable is not : check permissions")
        args = [exe] 
        args += self._command_out(rundir)
        args += self._format_args(rundir, **params_kw)

        # prepare modified command-line arguments with appropriate format
        for p in self.params(**params_kw):
            args += self._command_param(p.name, p.value)

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

    #def load(self, file=None, rundir=None, reload=False):
    #    """
    #    reload : if True, reload params from rundir if when it is present in run.json
    #    """
    #    cfg = json.load(open(file or cls.filename(rundir)))
    #    kwargs = cfg['params']
    #    if not cfg['state'] or reload:
    #        kwargs.update(self.readstate(cfg["rundir"]))
    #    else:
    #        kwargs.update(cfg['state'])
    #    return self(cfg['rundir'], **kwargs)

    def todict(self, rundir, params_kw, state_kw=None):
        return odict([
            ('rundir', rundir), 
            ('params', params_kw),
            ('state', state_kw or odict()),
            ('command', self.command(rundir, params_kw)),
            ('inidir', self.work_dir.format(rundir)),
            ('environ', self.environ(rundir, params_kw)),
            ('time', str(datetime.datetime.now())),
            ('version', __version__),
        ])


    def run(self, rundir, params_kw, background=True, submit=False, **kwargs):
        """Run the model (background subprocess, submit to slurm...)
        """
        # create run directory
        if not os.path.exists(rundir):
            os.makedirs(rundir)

        frozenmodel = self(rundir, **params_kw)
        frozenmodel.save()

        # write param file to rundir
        if self.filetype and self.filename:
            #TODO: have model params as a dictionary
            #TODO: rename filename --> file_in OR file_param
            filepath = os.path.join(self.rundir, self.filename)
            paramlist = frozenmodel.params
            self.filetype.dump(paramlist, open(filepath, 'w'))

        # prepare
        env = self.environ(rundir, params_kw, 
                           env=None if submit else os.environ.copy()) 
        args = self.command(rundir, params_kw)
        workdir = self.work_dir.format(rundir)
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
                p = subprocess.Popen(args, env=env, cwd=workdir, 
                                     stdout=stdout, stderr=stderr)
            except OSError as error:
                raise OSError("FAILED TO EXECUTE: `"+" ".join(args)+"` FROM `"+workdir+"`")

        frozenmodel.process = p

        if not background:
            ret = frozenmodel.wait()

        return frozenmodel


    ## Load model state
    def readstate(self, rundir):
        """get model state from original output (return dict)
        """
        if not self.filename_output:
            return {}
        assert self.filetype_output, "filetype_output is required"
        variables = self.filetype_output.load(open(os.path.join(rundir, self.filename_output)))
        return odict([(v.name,v.value) for v in variables])


    @classmethod
    def runfile(cls, rundir):
        return os.path.join(rundir, "run.json")

    def load(self, rundir):
        " load model state from output directory "
        cfg = json.load(open(self.runfile(rundir)))
        kwds = cfg["params"]
        kwds.update( cfg["state"] )
        return self(rundir, **kwds)


    def __call__(self, *args, **kwargs):
        """freeze model with rundir and params
        """
        return FrozenModel(self, *args, **kwargs)



class FrozenModel(object):
    """'Frozen' model instance representing a model run, with fixed rundir, params and state variables
    """
    def __init__(self, *args, **kwds):
        """
        *args : variable arguments with at least (model, rundir)
        **kwds : key-word arguments, parameter

        Example : FrozenModel(model, rundir, a=2, b=2, output=4)
        """
        assert len(args) == 2, 'expected 2 arg (model, rundir) got '+str(args)
        self.args = args
        self.kwds = kwds
        self.process = None  # with wait method

    @property
    def model(self):
        return self.args[0]

    @property
    def rundir(self):
        return self.args[1]

    def prior(self):
        """prior parameter distribution
        """
        return self.model.params(**self.kwds)

    def likelihood(self):
        """state variables' likelihood
        """
        like = self.model.state(**self.kwds)

    def posterior(self):
        """model posterior (params' prior * state posterior)
        """
        return self.prior + self.likelihood

    @property
    def params(self):
        return self.prior.as_dict()

    @property
    def state(self):
        return self.likelihood.as_dict()

    def todict(self):
        return self.model.todict(self.rundir, self.kwds)

    def save(self, file=None):
        # write summary to file
        cfg = self.todict()
        with open(file or self.model.runfile(rundir), 'w') as f:
            json.dump(cfg, f, 
                      indent=2, 
                      default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    def load(self):
        " load model state from output directory "
        cfg = json.load(open(self.model.runfile(self.rundir)))
        self.kwds.update(cfg["params"])
        self.kwds.update(cfg["state"])
        return self

    def readstate(self):
        self.kwds.update(self.model.readstate(self.rundir))
        return self

    def postproc(self):
        self.readstate().save()
        return self

    def wait(self):
        if self.process is not None:
            ret = self.process.wait()
        else:
            ret = None
        self.postproc()
        return ret



class CustomModel(Model):
    """User-provided model (e.g. via job install)
    """
    def __init__(self, command=None, setup=None, getvar=None, getcost=None, **kwargs):
        self._command = command
        self._setup = setup
        self._getvar = getvar
        self._getcost = getcost
        super(CustomModel, self).__init__(**kwargs)

    def setup(self, rundir, ):
        super(CustomModel, self).setup(rundir)  # write metadata
        if self._setup is not None:
            self._setup(rundir, self.executable, *self._format_args(rundir), **self.params.as_dict())

    def command(self, rundir):
        if self._command is None:
            return super(CustomModel, self).command(rundir)
        return self._command(rundir, self.executable, *self._format_args(rundir), **self.params.as_dict())

    def getvar(self, name, rundir):
        if self._getvar is None:
            return super(CustomModel, self).getvar(name, rundir)
        return self._getvar(name, rundir)

    def getcost(self, rundir):
        if self._getcost is None:
            return super(CustomModel, self).getcost(rundir)
        try:
            return self._getcost(rundir, self.executable, *self._format_args(rundir))
        except TypeError:
            return self._getcost(rundir)

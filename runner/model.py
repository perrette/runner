from __future__ import print_function, absolute_import
import subprocess
import os
import logging
import sys
import json
import datetime
from collections import OrderedDict as odict
from runner.tools import parse_val
from runner.submit import submit_job
from runner import __version__
from runner.compat import basestring
#from runner.model.generic import get_or_make_filetype

# default values
ENV_OUT = "RUNDIR"


class Param(object):
    """default parameter --> useful to specify custom I/O formats
    """
    def __init__(self, name, default=None, help=None, value=None, **kwargs):
        """
        name : parameter name
        default : default value, optional
        help : help (e.g. to provide for argparse), optional
        **kwargs : any other attribute required for custom file formats
            or to define prior distributions.
        """
        self.name = name
        self.default = default
        self.value = value if value is not None else default
        self.help = help
        self.__dict__.update(kwargs)

    #def __repr__(self):
    #    return "{cls}(name={name},default={default},value={value})".format(cls=type(self).__name__, **self.__dict__)

    def __str__(self):
        return "{name}={value}".format(name=self.name, value=self.value)

    @classmethod
    def parse(cls, string):
        name, value = string.split('=')
        return cls(name, parse_val(value))

    def tojson(self, **kwargs):
        return json.dumps(self.__dict__)
        #return json.dumps(str(self))


def _update_params(params, params_kw, strict=False):
    """update a list of params with a dict
    """
    names = [p.name for p in params]

    if not isinstance(params_kw, dict):
        raise TypeError('update params/state :: expected dict, got: {}'.format(type(params_kw).__name__))

    for name in params_kw:
        value = params_kw[name]

        # update existing parameter
        if name in names:
            i = names.index(name)
            params[i].value = value

        # if no parameter found, depends on `strict`
        else:
            if strict:
                import difflib
                logging.error("Available parameters:"+" ".join(names))
                suggestions = difflib.get_close_matches(name, names)
                if suggestions:
                    logging.error("Did you mean: "+ ", ".join(suggestions)+ " ?")
                raise ValueError("unknown parameter:"+repr(name))

            else:
                params.append(Param(name, value=value))


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
            list of model parameters to be updated with modified params
        * state : [Param], optional
            list of model state variables (output)
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
        if isinstance(args, basestring):
            args = args.split()
        self.args = args or []
        self.params = params or []
        self.state = state or []
        self.filetype = filetype
        self.filename = filename
        self.filetype_output = filetype_output
        self.filename_output = filename_output
        self.arg_out_prefix = arg_out_prefix
        self.arg_param_prefix = arg_param_prefix
        self.env_prefix = env_prefix
        self.env_out = env_out
        self.context = dict(filename=filename, 
                            executable=executable,
                            runid = '{runid}',
                            ) # for formatting args and environment variable etc.
        self.work_dir = work_dir

        # check !
        if filename:
            if filetype is None: 
                raise ValueError("need to provide FileType with filename")
            if not hasattr(filetype, "dumps"):
                raise TypeError("invalid filetype: no `dumps` method: "+repr(filetype))


    def update(self, params=None, state=None, context=None, strict=False):
        """Update parameters and state variables
        """
        _update_params(self.params, params or {}, strict)
        _update_params(self.state, state or {}, strict)
        self.context.update(context or {})


    def save(self, rundir, cfg=None):
        """save model state and parameter
        """
        cfg = cfg or {
            'time': str(datetime.datetime.now()),
            'version': __version__,
            'params': {p.name:p.value for p in self.params},
            'state': {p.name:p.value for p in self.state},
            'workdir': os.getcwd(),
            'executable': self.executable,
            'args': self._format_args(rundir),
            'sys.argv': sys.argv,
        }
        json.dump(cfg, open(os.path.join(rundir, "run.json"), 'w'), 
                  sort_keys=True, indent=2, default=lambda x:x.tolist() if hasattr(x, 'tolist') else x )


    def setup(self, rundir):
        """create run directory and write parameters to file
        """
        if not os.path.exists(rundir):
            os.makedirs(rundir)
        
        self.save(rundir)

        # for communication with the model
        if self.filetype and self.filename:
            #TODO: have model params as a dictionary
            filepath = os.path.join(rundir, self.filename)
            self.filetype.dump(self.params, open(filepath, 'w'))

    def _command_out(self, rundir):
        if self.arg_out_prefix is None:
            return []
        return (self.arg_out_prefix + rundir).split() 

    def _command_param(self, name, value, **kwargs):
        if self.arg_param_prefix is None:
            return []
        prefix = self.arg_param_prefix.format(name, name=name, **kwargs)
        return (prefix + str(value)).split()

    def _format_args(self, rundir):
        """two-pass formatting: first rundir and params with `{}` and `{NAME}`
        then context `{{rundir}}` `{{runid}}`
        """
        paramskw = self.params_as_dict()
        return [arg.format(rundir, **paramskw).format(rundir=rundir, **self.context) 
                for arg in self.args]

    def command(self, rundir):
        exe = self.executable
        if exe is None:
            raise ValueError("model requires an executable")
        elif os.path.isfile(exe):
            if not os.access(exe, os.X_OK):
                raise ValueError("model executable is not : check permissions")
        args = [exe] 
        args += self._command_out(rundir)
        args += self._format_args(rundir)

        # prepare modified command-line arguments with appropriate format
        for p in self.params:
            args += self._command_param(**p.__dict__)

        return args

    def params_as_dict(self):
        return {p.name:p.value for p in self.params}

    def environ(self, rundir, env=None):
        """define environment variables to pass to model
        """
        if self.env_prefix is None:
            return None

        # prepare variables to pass to environment
        context = self.context.copy()
        if self.env_out is not None:
            context[self.env_out] = rundir 
        context.update(self.params_as_dict())

        # format them with appropriate prefix
        update = {self.env_prefix+k:str(context[k])
               for k in context if context[k] is not None}

        # update base environment
        env = env or {}
        env.update(update)

        return env


    def run(self, rundir, background=True):
        """open subprocess and return Popen instance 
        """
        self.setup(rundir)
        env = self.environ(rundir, env=os.environ.copy()) 
        args = self.command(rundir)

        if background:
            stdout = open(os.path.join(rundir, 'log.out'), 'w')
            stderr = open(os.path.join(rundir, 'log.err'), 'w')
        else:
            stdout = None
            stderr = None

        workdir = self.work_dir.format(rundir) if self.work_dir else "."

        try:
            p = subprocess.Popen(args, env=env, cwd=workdir, 
                                 stdout=stdout, stderr=stderr)
        except OSError as error:
            #if os.path.isfile(args[0]) and not args[0].startswith('.'):
            #    print("Check executable name (use leading . or bash)")
            #raise OSError("NOT EXECUTABLE: "+" ".join(args))
            raise OSError("FAILED TO EXECUTE: `"+" ".join(args)+"` FROM `"+workdir+"`")

        if not background:
            ret = p.wait()

        return p


    def submit(self, rundir, jobfile=None, output=None, error=None, **kwargs):
        """Submit job to slurm or whatever is specified via **kwargs
        """
        self.setup(rundir)
        env = self.environ(rundir)
        args = self.command(rundir)
        output = output or os.path.join(rundir, "log.out")
        error = error or os.path.join(rundir, "log.err")
        jobfile = jobfile or os.path.join(rundir, 'submit.sh')
        workdir = self.work_dir.format(rundir) if self.work_dir else "."
        return submit_job(" ".join(args), env=env, workdir=workdir, 
                          output=output, error=error, jobfile=jobfile, **kwargs)

    ## POST-processing
    def _readstate(self, rundir):
        """get model state from original output (return dict)
        """
        if not self.filename_output:
            return {}
        logging.info('read state')
        assert self.filetype_output, "filetype_output is required"
        variables = self.filetype_output.load(open(os.path.join(rundir, self.filename_output)))
        return odict([(v.name,v.value) for v in variables])


    def load(self, rundir, reload=False, save=False):
        """load state variable and params from run directory

        reload: bool, if True, reload from model own param file instead of run.json
        save: bool, if True, save state to run.json file
        """
        # was written upon run
        fname = os.path.join(rundir, "run.json")
        cfg = json.load(open(fname))
        if 'state' not in cfg:
            cfg['state'] = {} # back compat

        # load state variables
        if not cfg['state'] or reload:
            cfg['state'] = self._readstate(rundir)
            if save:
                self.save(rundir, cfg)

        self.update(cfg['params'], cfg['state'])


    def getvar(self, name, rundir):
        """get state variable by name given run directory
        """
        if not self.state:
            self.load(rundir)
        names = [p.name for p in self.state]
        try:
            i = names.index(name)
        except ValueError: 
            logging.error("Available parameters:"+(" ".join(names) if names else "None"))
            raise
        return self.state[i].value


    def getcost(self, rundir):
        """get cost-function for one member (==> weight = exp(-0.5*cost))
        """
        raise NotImplementedError("getcost")



class CustomModel(Model):
    """User-provided model (e.g. via job install)
    """
    def __init__(self, command=None, setup=None, getvar=None, getcost=None, **kwargs):
        self._command = command
        self._setup = setup
        self._getvar = getvar
        self._getcost = getcost
        super(CustomModel, self).__init__(**kwargs)

    def setup(self, rundir):
        super(CustomModel, self).setup(rundir)  # write metadata
        if self._setup is not None:
            self._setup(rundir, self.executable, *self._format_args(rundir), **self.params_as_dict())

    def command(self, rundir):
        if self._command is None:
            return super(CustomModel, self).command(rundir)
        return self._command(rundir, self.executable, *self._format_args(rundir), **self.params_as_dict())

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

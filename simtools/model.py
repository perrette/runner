from __future__ import print_function, absolute_import
import difflib
import subprocess
import os
import json
from simtools.tools import parse_val
from simtools.submit import submit_job
#from simtools.model.generic import get_or_make_filetype

ARG_TEMPLATE = "--{name} {value}" # by default 
OUT_TEMPLATE = "--out {rundir}" # by default 

ENV_PREFIX = "SIMTOOLS_"
ENV_RUNDIR = "RUNDIR"



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
    def __init__(self, executable=None, args=None, params=None, filetype=None, filename=None, out_template=OUT_TEMPLATE, arg_template=ARG_TEMPLATE, env_out=ENV_RUNDIR, env_prefix=ENV_PREFIX):
        """
        * executable : runscript
        * args : [str] or str, optional
            list of command arguments to pass to executable. They may contain
            formattable patterns {rundir}, {runid}, {runtag}. Typically run directory
            or input parameter file. Prefer out_template for output directory.
        * params : [Param], optional
            list of model parameters to be updated with modified params
            If params is provided, strict checking of param names is performed during 
            update, with informative error message.
        * filetype : ParamsFile instance or anything with `dump` method, optional
        * filename : relative path to rundir, optional
            filename for parameter passing to model (also needs filetype)
        * out_template : str, optional
            command-line passing of output dir, e.g. "--out {}" or "{}"
        * arg_template : str, optional
            command-line passing of one parameter, e.g. "--{} {}" (name, value)
        * env_out : str, optional
            environment variable name for output directory (to be appended to prefix)
        * env_prefix : str, optional
            environment passing of parameters, e.g. "SIMTOOLS_" to be completed
            with parameter name or RUNDIR for model output directory.
        """
        self.executable = executable
        if isinstance(args, basestring):
            args = args.split()
        self.args = args or []
        self.params = params or []
        self.filetype = filetype
        self.filename = filename
        self.out_template = out_template or ""
        self.arg_template = arg_template or ""
        self.env_prefix = env_prefix or ""
        self.env_out = env_out or ""
        self.context = dict(filename=filename, executable=executable) # for formatting args and environment variable etc.

        self.strict = len(self.params > 0)

        # check !
        if filename:
            if filetype is None: 
                raise ValueError("need to provide FileType with filename")
            if not hasattr(filetype, "dumps"):
                raise TypeError("invalid filetype: no `dumps` method: "+repr(filetype))


    def update(self, params_kw, context=None):
        """Update parameter from ensemble
        """
        names = [p.name for p in self.params]

        for name in params_kw:
            value = params_kw[name]

            # update existing parameter
            if name in names:
                i = names.index(name)
                self.params[i].value = value

            # if no parameter found, never mind, may check or not
            else:
                if self.strict:
                    print("Available parameters:"," ".join(names))
                    suggestions = difflib.get_close_matches(name, names)
                    if suggestions:
                        print("Did you mean: ", ", ".join(suggestions), "?")
                    raise ValueError("unknown parameter:"+repr(name))
                else:
                    self.params.append(Param(name, value=value))

        self.context.update(context or {})


    def setup(self, rundir):
        """create run directory and write parameters to file
        """
        if not os.path.exists(rundir):
            os.makedirs(rundir)
        if self.filetype and self.filename:
            #TODO: have model params as a dictionary
            self.filetype.dump(self.params, open(self.filename, 'w'))

    @staticmethod
    def _command_out(rundir):
        return self.out_template.format(rundir, rundir=rundir).split()

    @staticmethod
    def _command_param(name, value, **kwargs):
        return self.arg_template.format(name, value, name=name, value=value, **kwargs).split()

    def _format_args(self, rundir):
        return [arg.format(rundir, rundir=rundir, **self.context) for arg in self.args]

    def command(self, rundir):
        if self.executable is None:
            raise ValueError("model requires an executable")
        args = [self.executable] 
        args += self._command_out(rundir)
        args += self._format_args(rundir, **self.context)

        # prepare modified command-line arguments with appropriate format
        for p in self.params:
            args += self._command_param(p.name, p.value, **p.__dict__)

        return args

    def params_as_dict(self):
        return {p.name:p.value for p in self.params}

    def environ(self, rundir):
        """define environment variables to pass to model
        """
        # prepare variables to pass to environment
        context = self.context.copy()
        if self.env_out:
            context[self.env_out] = rundir 
        context.update(self.params_as_dict())

        # format them with appropriate prefix
        env = {self.env_prefix.upper()+k.upper():context[k] 
               for k in context if context[k] is not None}
        return env


    def run(self, rundir, **kwargs):
        """open subprocess
        """
        env = os.environ.copy()
        env.update( self.environ(rundir) )
        args = self.command(rundir)
        return subprocess.Popen(args, env=env, **kwargs)

    def submit(self, rundir, **kwargs):
        """Submit job to slurm or whatever is specified via **kwargs
        """
        env = self.environ(rundir)
        args = self.command(rundir)
        return submit_job(" ".join(args), env=env, **kwargs)

    def getvar(self, name, rundir):
        """get state variable by name given run directory
        """
        raise NotImplementedError("getvar")



class CustomModel(Model):
    """User-provided model (e.g. via job install)
    """
    def __init__(self, command=None, setup=None, getvar=None, **kwargs):
        self._command = command
        self._setup = setup
        self._getvar = getvar
        super(CustomModel, self).__init__(**kwargs)

    def setup(self, rundir):
        if self._setup is None:
            return super(CustomModel, self).setup(rundir)
        self._setup(rundir, *self._format_args(rundir), **self.params_as_dict())

    def command(self, rundir):
        if self._command is None:
            return super(CustomModel, self).command(rundir)
        return self._command(rundir, *self._format_args(rundir), **self.params_as_dict())

    def getvar(self, name, rundir):
        if self._getvar is None:
            raise ValueError("no getvar function provided")
        return self._getvar(name, rundir)

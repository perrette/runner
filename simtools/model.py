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
    def __init__(self, executable=None, args=None, params=None, filetype=None):
        self.executable = executable
        self.args = args
        self.params = params or []
        self.filetype = filetype
        self.strict = len(self.params > 0)

    def setup(self, rundir, context=None):
        pass
        #raise NotImplementedError("setup")

    def command(self, rundir, context=None):
        raise NotImplementedError("command")

    def run(self, rundir, context=None, **kwargs):
        """Popen(Model.command(context), **kwargs)
        """
        args = self.command(rundir, context)
        return subprocess.Popen(args, **kwargs)

    def submit(self, rundir, context=None, **kwargs):
        """Slurm(Model.command(context), **kwargs)
        """
        args = self.command(rundir, context)
        return submit_job(" ".join(args), **kwargs)

    def getvar(self, name, rundir):
        raise NotImplementedError("getvar")


    def update(self, params_kw):
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


    def params_as_dict(self):
        return {p.name:p.value for p in self.params}



class GenericModel(Model):
    """Generic model configuration
    """
    def __init__(self, executable, args=None, params=None, 
                 arg_template=ARG_TEMPLATE, out_template=OUT_TEMPLATE, 
                 filename=None, filetype=None, setenviron=False):
        """
        executable : runscript
        args : [str] or str, optional
            list of command arguments to pass to executable. They may contain
            formattable patterns {rundir}, {runid}, {runtag}. Typically run directory
            or input parameter file.
        params : [Param], optional
            list of model parameters to be updated with modified params
            If params is provided, strict checking of param names is performed during 
            update, with informative error message.
        arg_template : str, optional
            Indicate parameter format for command-line with placeholders `{name}` and 
            `{value}`, or empty placeholders {} in this order. 
            By default `--{name} {value}`, but note that any field
            in parameter definition can be used. Set to None or empty list to avoid
            passing parameters via the command-line.
        out_template : str, optional
            template to pass  the output directory to the model, use {} or {rundir}
        filename : str, optional
            By default parameters are provided as command-line arguments but if file
            name is provided they will be written to file. Path is relative to rundir.
        filetype : ParamsFile instance or anything with `dump` method, optional
        setenviron : bool, optional
            if True, define parameter values as environment variables
        """
        self.executable = executable
        if isinstance(args, basestring):
            args = args.split()
        self.args = args or []
        self.params = params or []
        self.strict = len(self.params) > 0  
        if isinstance(arg_template, list):
            arg_template = " ".join(arg_template)
        self.arg_template = arg_template
        if isinstance(out_template, list):
            out_template = " ".join(out_template)
        self.out_template = out_template 
        self.filename = filename 
        self.filetype = filetype
        self.setenviron = setenviron

        if filename:
            if filetype is None: 
                raise ValueError("need to provide FileType with filename")
            if not hasattr(filetype, "dumps"):
                raise TypeError("invalid filetype: no `dumps` method: "+repr(filetype))



    def setup(self, rundir):
        """Write param file to rundir if necessary
        """
        if not os.path.exists(rundir):
            os.makedirs(rundir)
        if self.filename:
            fname = os.path.join(rundir, self.filename)
            with open(fname, "w") as f:
                self.filetype.dump(self.params, f)


    def command(self, rundir=None, **context):
        """
        context : dict of experiment variables such as `rundir` and `runid`, which 
            maybe used to format some of the commands before passing to Popen.
        """
        if self.executable is None:
            raise ValueError("model requires an executable")
        args = [self.executable] + [self.out_template.format(rundir, rundir=rundir)] + self.args

        # prepare modified command-line arguments with appropriate format
        for p in self.params:
            string = self.arg_template.format(p.name, p.value, **p.__dict__)
            args.append(string.split())

        # replace patterns such as {runid} in command
        for i, arg in enumerate(args):
            args[i] = arg.format(rundir=rundir, **context)

        return args

    def getenv(self, rundir, context=None):
        env = makenv(context or {})
        env.update(self.params_as_dict())
        env.update({'rundir':rundir})
        return env

    def run(self, rundir, context=None, **kwargs):
        """Popen(Model.command(context), **kwargs)
        """
        if self.setenviron:
            env = os.environ.copy()
            env.update(self.getenv(rundir, context))
            kwargs['env'] = env
        super(GenericModel, self).run(rundir, context, **kwargs)

    def submit(self, rundir, context=None, **kwargs):
        if self.setenviron:
            kwargs['env'] = self.getenv(rundir, context)
        super(GenericModel, self).submit(rundir, context, **kwargs)


ENVPREFIX = "SIMTOOLS_"

def makenv(context, env=None):
    env = env or {}
    for k in context:
        if context[k] is not None:
            env[ENVPREFIX+k.upper()] = context[k]
    return env


class CustomModel(Model):
    """User-provided model (e.g. via job install)
    """
    def __init__(self, executable=None, command=None, setup=None, getvar=None, args=None, params=None, filetype=None):
        self._command = command
        self._setup = setup
        self._getvar = getvar
        super(CustomModel, self).__init__(executable, args, params, filetype=filetype)

    def setup(self, rundir, context=None):
        if self._setup is None:
            return  # no setup ...
        self._setup(rundir)

    def command(self, rundir, context=None):
        if self._command is None:
            raise ValueError("no make_command function provided")
        return self._command(rundir, 
                             self.args, 
                             **{p.name:p.value for p in self.params})

    def getvar(self, name, rundir):
        if self._getvar is None:
            raise ValueError("no getvar function provided")
        return self._getvar(name, rundir)

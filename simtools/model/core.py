from __future__ import print_function, absolute_import
import difflib
import subprocess
import os
import json

from simtools.submit import submit_job
from simtools.model.params import Param, ParamsFile
from simtools.model.generic import get_or_make_filetype

PARAMS_ARG = "--{name} {value}" # by default 


# Model instance
# ==============
class Model(object):
    """Generic model configuration
    """
    def __init__(self, executable, args=None, params=None, params_args=PARAMS_ARG, params_write=None, filetype=None):
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
        params_args : [str] or str, optional
            Indicate parameter format for command-line with placeholders `{name}` and 
            `{value}`. By default `["--{name}", "{value}"]`, but note that any field
            in parameter definition can be used. Set to None or empty list to avoid
            passing parameters via the command-line.
        params_write : str, optional
            By default parameters are provided as command-line arguments but if file
            name is provided they will be written to file. Path is relative to rundir.
        filetype : ParamsFile instance or str or anything with `dump` method, optional
        """
        self.executable = executable
        if isinstance(args, basestring):
            args = args.split()
        self.args = args or []
        self.params = params or []
        self.strict = len(self.params) > 0  
        if isinstance(params_args, basestring):
            params_args = params_args.split()
        self.params_args = params_args or []
        self.params_write = params_write 
        self.filetype = get_or_make_filetype(filetype)

    @classmethod
    def read(cls, configfile, root='model'):
        with open(configfile) as f:
            dat = json.load(f)
        if root:
            dat = dat[root]
        return cls.fromconfig(dat)


    @classmethod
    def fromconfig(cls, dat):
        """Initialize Model from dictionary config
        """
        dat = dat.copy()

        pdef = dat.pop("params", {})

        write = pdef.pop("write", None)
        args = pdef.pop("args", PARAMS_ARG)
        default = pdef.pop("default", None)
        filetypedat = pdef.pop("file", None)
        filetype = get_or_make_filetype(filetypedat)

        # read default params
        if default is not None:
            if isinstance(default, basestring):
                params = filetype.load(open(default))

            elif isinstance(default, dict):
                params = [Param(k, default[k]) for k in default]

            else:
                raise ValueError("invalid format for default params:"+repr(default))
        else:
            params = []

        # Initialize model class
        dat.update(dict(
            params=params,
            filetype=filetype,
            params_write=write,
            params_args=args,
        ))

        return cls(**dat)


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


    def setup(self, rundir):
        """Write param file to rundir if necessary
        """
        if not os.path.exists(rundir):
            os.path.makedirs(rundir)
        if self.params_write:
            fname = os.path.join(rundir, self.params_write)
            with open(fname, "w") as f:
                self.filetype.dump(self.params, f)


    def command(self, context=None):
        """
        context : dict of experiment variables such as `expdir` and `runid`, which 
            maybe used to format some of the commands before passing to Popen.
        """
        args = [self.executable] + self.args

        # prepare modified command-line arguments with appropriate format
        for p in self.params:
            for c in self.params_args:
                args.append(c.format(**p.__dict__))

        # format command string with `rundir`, `runid` etc.
        context = context or {}

        # replace patterns such as {runid} in command
        for i, arg in enumerate(args):
            args[i] = arg.format(**context)

        return args


    def run(self, context=None, **kwargs):
        """Popen(Model.command(context), **kwargs)
        """
        args = self.command(context)
        return subprocess.Popen(args, **kwargs)


    def submit(self, context=None, **kwargs):
        """Slurm(Model.command(context), **kwargs)
        """
        args = self.command(context)
        return submit_job(" ".join(args), **kwargs)

from __future__ import print_function, absolute_import
import difflib
import subprocess
import os
import json

from simtools.submit import submit_job
from simtools.model.params import Param, ParamsFile
#from simtools.model.generic import get_or_make_filetype

PARAMS_ARG = "--{name} {value}" # by default 


# Model instance
# ==============
class Model(object):
    """Generic model configuration
    """
    def __init__(self, executable, args=None, params=None, arg_template=PARAMS_ARG, filename=None, filetype=None):
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
        arg_template : [str] or str, optional
            Indicate parameter format for command-line with placeholders `{name}` and 
            `{value}`. By default `["--{name}", "{value}"]`, but note that any field
            in parameter definition can be used. Set to None or empty list to avoid
            passing parameters via the command-line.
        filename : str, optional
            By default parameters are provided as command-line arguments but if file
            name is provided they will be written to file. Path is relative to rundir.
        filetype : ParamsFile instance or anything with `dump` method, optional
        """
        self.executable = executable
        if isinstance(args, basestring):
            args = args.split()
        self.args = args or []
        self.params = params or []
        self.strict = len(self.params) > 0  
        if isinstance(arg_template, basestring):
            arg_template = arg_template.split()
        self.arg_template = arg_template or []
        self.filename = filename 
        self.filetype = filetype

        if filename:
            if filetype is None: 
                raise ValueError("need to provide FileType with filename")
            if not hasattr(filetype, "dumps"):
                raise TypeError("invalid filetype: no `dumps` method: "+repr(filetype))

        #if not hasattr(filetype, "dumps") 
        #self._check_paramsio()
        #self.filetype = get_or_make_filetype(filetype)

    #def _check_paramsio(self):
    #    """check default params or possibly read from file
    #    """
    #    if not hasattr(self.filetype, 'dumps'):
    #        self.filetype = get_or_make_filetype(self.filetype)

    #    # read default params
    #    if not isinstance(self.params, list):
    #        if isinstance(self.params, basestring):
    #            self.params = self.filetype.load(open(self.params))

    #        elif isinstance(self.params, dict):
    #            self.params = [Param(k, self.params[k]) for k in self.params]

    #        elif self.params is None:
    #            self.params = []

    #        else:
    #            raise ValueError("invalid format for params_default:"+repr(self.params))
    #    else:
    #        for p in self.params:
    #            if not hasattr(p, 'name') or not hasattr(p, 'value'):
    #                raise TypeError('model params have wrong type:'+repr(p))


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
        if self.filename:
            fname = os.path.join(rundir, self.filename)
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
            for c in self.arg_template:
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

"""Model definition, mostly Param I/O
"""
from __future__ import print_function, absolute_import
import os
import json
import difflib
from simtools.addons import filetypes, register_filetype
import subprocess
from .submit import submit_job

#PARAMS_ARG = ["--{name}", "{value}"] # for command line
PARAMS_ARG = None # for command line
PARAMS_FILE_TYPE = "jsondict" # {"jsondict", "jsonlist", "template", "generic", "nml"}
PARAMS_FILE_GENERIC = "{name}={value}"


# Parameters I/O to communicate with model
# ========================================
class Param(object):
    """default parameter --> useful to specify custom I/O formats
    """
    def __init__(self, name, default=None, help=None, value=None, **kwargs):
        """
        name : parameter name
        default : default value, optional
        help : help (e.g. to provide for argparse), optional
        **kwargs : any other attribute required for custom file formats
        """
        self.name = name
        self.default = default
        self.value = value if value is not None or default
        self.help = help
        self.__dict__.update(kwargs)

    def __repr__(self):
        return "Param(name={name},default={default},value={value})".format(**self.__dict__)

    def __str__(self):
        return "{name}={value} [{default}]".format(**self.__dict__)


class ParamsFile(object):
    def dumps(self, params):
        raise NotImplementedError()

    def loads(self, string):
        raise NotImplementedError()

    def dump(self, params, f):
        f.write(self.dumps(params))

    def load(self, f):
        return self.loads(f.read())


class JsonList(ParamsFile):
    """json file format
    """
    def __init__(self, indent=2, **kwargs):
        kwargs["indent"] = indent
        self.kwargs = kwargs

    def dumps(self, params):
        return json.dumps([p.__dict__ for p in params], **kwargs)

    def loads(self, string):
        return [Param(**p) for p in json.loads(string)]


class JsonDict(ParamsFile):
    """json file format
    """
    def __init__(self, indent=2, sort_keys=True, **kwargs):
        kwargs["indent"] = indent
        kwargs["sort_keys"] = sort_keys
        self.kwargs = kwargs

    def dumps(self, params):
        return json.dumps({p.name:p.value for p in params}, **kwargs)

    def loads(self, string):
        kwargs = json.loads(string)
        return [Param(name=k, value=kwargs[k]) for k in sorted(kwargs.keys())]


class GenericFile(ParamsFile):
    """Generic class to write to a parameter file
    """
    def __init__(self,  line_fmt=PARAMS_FILE_GENERIC):
        """
        line_fmt : str, optional
            param format for each line, with placeholders {name} and {value}.
            By default "{name}={value}".
        """
        self.line_fmt = line_fmt

    def dumps(self, params):
        """return the file as a string
        """
        lines = [ self.line_fmt.format(**p.__dict__) for p in params ]
        return "\n".join(lines)


class TemplateFile(ParamsFile):
    """Custom file format based on a full file template
    """
    def __init__(self, string):
        assert string, "must provide template string"
        self.string = string

    def dumps(self, params):
        return self.string.format(**{p.name:p.value for p in params})

    @classmethod
    def read(cls, file):
        return cls(open(file).read())


register_filetype("jsonlist", JsonList)
register_filetype("jsondict", JsonDist)
register_filetype("generic", GenericFile)
register_filetype("template", TemplateFile.read)


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
        filetype : ParamsFile instance or anything with `dump` method, optional
            By default JsonDict.
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
        self.filetype = filetype or JsonDict()
        if not hasattr(self.filetype, "dump"):
            raise TypeError("filetype has no `dump` method: "\
                            +repr(self.filetype))

    @classmethod
    def read(cls, configfile, root=None):
        with open(configfile) as f:
            dat = json.load(f)
        if root:
            dat = dat[root]
        return cls.fromdict(dat)

    @classmethod
    def fromdict(cls, dat):
        """Initialize from dictionary config
        """
        dat = dat.copy()

        pdef = dat.pop("params", {})

        type_name = pdef.pop("filetype", PARAMS_FILE_TYPE)
        type_def = pdef.pop("filetype_def", {}) # key-word arguments
        write = pdef.pop("write", None)
        args = pdef.pop("args", PARAMS_ARG)
        default = pdef.pop('default', None)

        # param file def?
        if type_name in filetypes:
            cls = filetypes[type_name]
            filetype = cls(**type_def)

        else:
            print("Available filetypes:",filetypes.keys())
            raise ValueError("Unknown file type: "+repr(type_name))

        # read default params
        if isinstance(default, basestring):
            params = filetype.load(open(default)))

        elif isinstance(default, dict):
            params = [Param(k, default[k]) for k in default]

        else:
            raise ValueError("invalid format for default params:"+repr(default))

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

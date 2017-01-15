"""Model definition, mostly Param I/O
"""
from __future__ import print_function, absolute_import
import os
import json
import difflib

#PARAM_CMD = ["--{name}", "{value}"] # for command line
PARAM_CMD = None # for command line
PARAMS_FILE_TYPE = "jsondict" # {"jsondict", "jsonlist", "template", "generic", "nml"}
PARAMS_FILE_GENERIC = "{name}={value}"


# dictionary of available file types
filetypes = {}

def register_filetype(name, cls, *args):
    """
    """
    filetypes[name] = cls, args

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
    def __init__(self,  param_fmt=PARAMS_FILE_GENERIC):
        """
        param_fmt : str, optional
            param format for each line, with placeholders {name} and {value}.
            By default "{name}={value}".
        """
        self.param_fmt = param_fmt

    def dumps(self, params):
        """return the file as a string
        """
        lines = [ self.param_fmt.format(**p.__dict__) for p in params ]
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


register_filetype("jsonlist", JsonList, 2)
register_filetype("jsondict", JsonDist, 2)
register_filetype("generic", GenericFile, PARAMS_FILE_GENERIC)
register_filetype("template", TemplateFile.read)


# Model instance
# ==============
class Model(object):
    """Generic model configuration
    """
    def __init__(self, executable, command=None, params=None, params_cmd=PARAM_CMD, params_write=None, params_filetype=None):
        """
        executable : runscript
        command : [str], optional
            list of command arguments to pass to executable. They may contain
            formattable patterns {rundir}, {runid}, {runtag}. Typically run directory
            or input parameter file.
        params : [Param], optional
            list of model parameters to be updated with modified params
            note this list is mostly useful to specify default values and to provide
            information related to specific file formats, but is not required in the 
            case of command-line arguments.
        params_cmd : [str], optional
            Indicate parameter format for command-line with placeholders `{name}` and 
            `{value}`. By default `["--{name}", "{value}"]`, but note that any field
            in parameter definition can be used. Set to None or empty list to avoid
            passing parameters via the command-line.
        params_write : str, optional
            By default parameters are provided as command-line arguments but if file
            name is provided they will be written to file. Path is relative to rundir.
        params_filetype : ParamsFile instance or anything with `dump` method, optional
            By default JsonDict.
        """
        self.executable = executable
        if isinstance(command, basestring):
            command = command.split()
        self.command = command or []
        self.params = params or []
        if isinstance(params_cmd, basestring):
            params_cmd = params_cmd.split()
        self.params_cmd = params_cmd or []
        self.params_write = params_write 
        self.params_filetype = params_filetype or JsonDict()
        if not hasattr(self.params_filetype, "dump"):
            raise TypeError("params_filetype has no `dump` method: "\
                            +repr(self.params_filetype))

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

        file_type = pdef.pop("filetype", PARAMS_FILE_TYPE)
        write = pdef.pop("write", None)
        command = pdef.pop("command", PARAM_CMD)
        default = pdef.pop('default', None)

        # param file def?
        if file_type in filetypes:
            cls, args = filetypes[file_type]
            # assume argument are located in a variable with same name as file type
            args = pdef.pop(file_type, args)
            filetype = cls(*args)

        else:
            print("Available filetypes:",filetypes.keys())
            raise ValueError("Unknown file type: "+repr(file_type))

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
            params_filetype=filetype,
            params_write=write,
            params_cmd=command,
        ))

        return cls(**dat)


    def update(self, params_kw, check=False):
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
                if check:
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
                self.params_filetype.dump(self.params, f)


    def make_command(self, context=None):
        """
        context : dict of experiment variables such as `expdir` and `runid`, which 
            maybe used to format some of the commands before passing to Popen.
        """
        args = [self.executable] + self.command

        # prepare modified command-line arguments with appropriate format
        for p in self.params:
            for c in self.params_cmd:
                args.append(c.format(**p.__dict__))

        # format command string with `rundir`, `runid` etc.
        context = context or {}

        # replace patterns such as {runid} in command
        for i, arg in enumerate(args):
            args[i] = arg.format(**context)

        return args

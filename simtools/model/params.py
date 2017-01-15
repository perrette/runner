"""Model definition, mostly Param I/O
"""
from __future__ import print_function, absolute_import
import json

FILETYPES = {}

def register_filetype(name, cls):
    FILETYPES[name] = cls


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
        self.value = value if value is not None else default
        self.help = help
        self.__dict__.update(kwargs)

    def __repr__(self):
        return "Param(name={name},default={default},value={value})".format(**self.__dict__)

    def __str__(self):
        return "{name}={value} [{default}]".format(**self.__dict__)


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

    DEFAULT = "jsondict"  # convenient to pack it here

    @classmethod
    def fromconfig(cls, dat=None):
        """Initialize ParamsFile from dictionary config
        """
        dat = dat or {}
        typename = dat.pop("type", cls.DEFAULT)
        settings = dat.pop("settings", {}) # key-word arguments

        # param file def?
        if typename in FILETYPES:
            cls = FILETYPES[typename]
            filetype = cls(**settings)

        else:
            print("Available filetypes:",FILETYPES.keys())
            raise ValueError("Unknown file type: "+repr(typename))

        return filetype


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


class GenericFile(ParamsFile):
    """Generic class to write to a parameter file
    """
    DEFAULT = "{name}={value}"

    def __init__(self,  line_fmt=DEFAULT):
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
register_filetype("jsondict", JsonDict)
register_filetype("generic", GenericFile)
register_filetype("template", TemplateFile.read)

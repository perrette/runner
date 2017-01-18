"""Parameters I/O to communicate with model
"""
from __future__ import print_function, absolute_import
import json

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

    #def __repr__(self):
    #    return "{cls}(name={name},default={default},value={value})".format(cls=type(self).__name__, **self.__dict__)

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


# Json file types
# ===============

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


# FileType register
# =================

filetypes = {}

def register_filetype(name, filetype):
    filetypes[name] = filetype

# register filetypes
register_filetype(None, JsonDict())  # the default
register_filetype("jsondict", JsonDict())
register_filetype("jsonlist", JsonList())


def get_filetype(name=None):
    """Return filetype instance based on string
    """
    if isinstance(name, ParamsFile) or hasattr(name, "dumps"):
        return name

    elif name in filetypes:
        return filetypes[name]

    else:
        raise ValueError("Unknown file type: "+repr(name))

def print_filetypes():
    print("Available filetypes:", ", ".join([repr(k) for k in filetypes.keys()]))

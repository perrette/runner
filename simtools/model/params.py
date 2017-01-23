from __future__ import print_function, absolute_import
import json
from simtools.tools import parse_val

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
        return json.dumps({p.name:p.value}, **kwargs)


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
        return json.dumps({p.name:p.value for p in params}, **self.kwargs)

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


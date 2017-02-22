"""Param type factory

Helper function to define your own file type.
Also take a look at runner.ext

For more complex formats you may want to define your own class. 
It takes subclassing `ParamsFile.dumps`, and if needed `ParamsFile.loads`.
"""
import json
from runner.model import ParamsFile, ParamIO
from runner.tools import parse_val

class FileTypeWrapper(ParamsFile):
    """take dict loads/dumps (json like), and make it work on params
    """
    def __init__(self, dumps=None, loads=None):
        self._loads = loads
        self._dumps = dumps

    def dumps(self, params):
        assert self._dumps is not None
        self._dumps({name:value for name,value in params})

    def loads(self, string):
        assert self._loads is not None
        kw = self._loads(string)
        return [ParamIO(k, kw[k]) for k in kw]


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
        return json.dumps({name:value for name,value in params}, **self.kwargs)

    def loads(self, string):
        kwargs = json.loads(string)
        return [ParamIO(name=k, value=kwargs[k]) for k in sorted(kwargs.keys())]


class TemplateFile(ParamsFile):
    """Custom file format based on a full file template (`dumps` ONLY)

    For example, for two parameters a and b:

    Here the parameter a : {a}
        {b} <-----  parameter b !
    """
    def __init__(self, template_file):
        self.template_file = template_file
        self._template = open(template_file).read()

    def dumps(self, params):
        return self._template.format(**{name:value for name,value in params})


class LineTemplate(ParamsFile):
    """Generic class with {name} and {value} placeholders (`dumps` ONLY !)

    Example:
    >>> filetype = LineTemplate("{name:>10}:{value:24}"))
    """
    def __init__(self,  line):
        self.line = line

    def dumps(self, params):
        lines = []
        for name,value in params:
            line = self.line.format(name, value, name=name, value=value)
            lines.append(line)
        return "\n".join(lines) + "\n"

class LineSeparator(LineTemplate):
    """Line-based format like "{name}{sep}{value}", `dumps` AND `loads`
    """
    def __init__(self, sep=None, reverse=False):
        self.sep = sep or " "
        self.reverse = reverse

    @property
    def line(self):
        line = "{name}"+self.sep+"{value}"
        if self.reverse:
            line = line.format(name="{value}", value="{name}")
        return line

    def loads(self, string):
        lines = string.splitlines()
        params = []
        for line in lines:
            name, value = line.split(self.sep.strip() or None)
            if self.reverse:
                name, value = value, name
            p = ParamIO(name.strip(), parse_val(value))
            params.append( p )
        return params

class LineSeparatorFix(LineSeparator):
    """Same as LineSeparator but with prefix and suffix
    """
    def __init__(self, sep=None, reverse=False, prefix="", suffix=""):
        LineSeparator.__init__(self, sep, reverse)
        self.prefix = prefix
        self.suffix = suffix

    @property
    def line(self):
        line = super(LineSeparatorFix, self)
        return self.prefix + line + self.suffix

    def loads(self, string):
        string = string.lstrip(self.prefix).rstrip(self.suffix)
        return LineSeparator.loads(self, string)


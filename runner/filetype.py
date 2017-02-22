"""Param type factory

Helper function to define your own file type.
Also take a look at runner.ext

For more complex formats you may want to define your own class. 
It takes subclassing `FileType.dumps`, and if needed `FileType.loads`.
"""
import json
from runner.tools import parse_val
from collections import OrderedDict as odict

class FileType(object):
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
class JsonFile(FileType):
    def dumps(self, params):
        return json.dumps(params, indent=2)+"\n"

    def loads(self, string):
        return json.loads(string)

class TemplateFile(FileType):
    """Custom file format based on a full file template (`dumps` ONLY)

    For example, for two parameters a and b:

    Here the parameter a : {a}
        {b} <-----  parameter b !
    """
    def __init__(self, template_file):
        self.template_file = template_file
        self._template = open(template_file).read()

    def dumps(self, params):
        return self._template.format(**params)


class LineTemplate(FileType):
    """Generic class with {name} and {value} placeholders (`dumps` ONLY !)

    Example:
    >>> filetype = LineTemplate("{name:>10}:{value:24}"))
    """
    def __init__(self,  line):
        self.line = line

    def dumps(self, params):
        lines = []
        for name,value in params.items():
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
            params.append( (name.strip(), parse_val(value)) )
        return odict(params)

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


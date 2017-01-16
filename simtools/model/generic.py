"""Param type factory

In addition to readily defined filetypes, you may indicate your own:

* one param per line, one-character separator. Simply indicate the separator.
    
    e.g. for "{name}|{value}" just indicate "|" as file type

* one param per line, general case. Provide a template using "{name}" and "{value}"
placeholders, such as the one above in full. Note this approach will be WRITE-ONLY, 
to pass parameters to your model (most useful anyway).

* use the `addon` system. Here with two lines in a `mytype.py` module:

        from simtools.model.generic import LineSeparator, register_filetype
        register_filetype("name|value", LineSeparator("|"))  # note: any name OK

    And edit your `config.json` with:
  
        addons = ["mytype"]
    
    For more complex formats you may want to define your own class. 
    It takes subclassing `ParamsFile.dumps`, and if needed `ParamsFile.load`.
    Take a look at `simtools.model.params.JsonDict` or 
    `simtools.model.generic.LineSeparator` to learn how to proceed.
"""
from simtools.model.params import ParamsFile, Param
from simtools.model.params import register_filetype, get_filetype, print_filetypes
from simtools.tools import parse_val


class TemplateFile(ParamsFile):
    """Custom file format based on a full file template (`dumps` ONLY)

    For example, for two parameters a and b:

    Here the parameter a : {a}
        {b} <-----  parameter b !
    """
    def __init__(self, string):
        assert string, "must provide template string"
        self.string = string

    def dumps(self, params):
        return self.string.format(**{p.name:p.value for p in params})

    @classmethod
    def read(cls, file):
        return cls(open(file).read())


class LineTemplate(ParamsFile):
    """Generic class with {name} and {value} placeholders (`dumps` ONLY !)
    """
    def __init__(self,  line):
        self.line = line

    def dumps(self, params):
        lines = []
        for p in params:
            line = self.line.format(**p.__dict__)
            lines.append(line)
        return "\n".join(lines)


class LineSeparator(LineTemplate):
    """Line-based format like "{name}{sep}{value}", `dumps` AND `loads`
    """
    def __init__(self, sep=None, reverse=False):
        self.sep = sep or " "
        self.line = "{name}"+self.sep+"{value}"
        self.reverse = reverse
        if reverse:
            self.line = self.line.format(name="{value}", value="{name}")

    def loads(self, string):
        lines = string.splitlines()
        params = []
        for line in lines:
            name, value = line.split(self.sep.strip() or None)
            if self.reverse:
                name, value = value, name
            p = Param(name.strip(), parse_val(value))
            params.append( p )
        return params


class LineSeparatorFix(LineSeparator):
    """Same as LineSeparator but with prefix and suffix
    """
    def __init__(self, sep=None, reverse=False, prefix="", suffix=""):
        LineSeparator.__init__(self, sep, reverse)
        self.prefix = prefix
        self.suffix = suffix
        self.line = self.prefix + self.line + self.suffix

    def loads(self, string):
        string = string.lstrip(self.prefix).rstrip(self.suffix)
        return LineSeparator.loads(self, string)


register_filetype("=", LineSeparator("="))
register_filetype(":", LineSeparator(":"))
register_filetype(",", LineSeparator(","))
register_filetype(";", LineSeparator(";"))
register_filetype(" ", LineSeparator(" "))


def get_or_make_filetype(name):
    """Note: this will be 
    """
    try:
        return get_filetype(name)
    except:
        pass

    # one --> separator?
    if len(name) == 1 or len(name.strip()) <= 1:
        return LineSeparator(name)

    # line template?
    try:
        name.format()
    except IndexError:
        # error means that brackets are present and therefore is a valid
        # LineTemplate filetype definition
        return LineTemplate(name)

    print(__doc__)
    print_filetypes()
    raise ValueError("Invalid file type: "+repr(name))

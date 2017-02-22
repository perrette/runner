"""Namelist parameter format

Originally adapted from 
https://github.com/leifdenby/namelist_python
"""
from __future__ import print_function, absolute_import
from collections import OrderedDict as odict
import re
from itertools import groupby
from runner.filetype import FileType
from runner.register import register_filetype

class ParamNml(object):
    def __init__(self, group, name, value, help=None):
        self.group = group
        self.name = name
        self.value = value
        self.help = help


class Namelist(FileType):
    """Namelist format
    """
    def __init__(self, sep='.'):
        self.sep = sep

    def dumps(self, params):
        pars = []
        for longname,value in params.items():
            group, name = longname.split(self.sep)
            pars.append(ParamNml(group, name, value))
        return format_nml(pars)

    def loads(self, string):
        params = parse_nml(string)
        return odict([(p.group + self.sep + p.name, p.value) for p in params])

register_filetype("namelist", Namelist(), '.nml')


def parse_nml(string, ignore_comments=False):
    """ parse a string namelist, and returns a list of param bundles
    with four attrs: name, value, help, group
    """
    group_re = re.compile(r'&([^&]+)/', re.DOTALL)  # allow blocks to span multiple lines
    array_re = re.compile(r'(\w+)\((\d+)\)')
    # string_re = re.compile(r"\'\s*\w[^']*\'")
    string_re = re.compile(r"[\'\"]*[\'\"]")
    # self._complex_re = re.compile(r'^\((\d+.?\d*),(\d+.?\d*)\)$')

    # list of parameters
    params = []
    # groups = odict()

    filtered_lines = []
    for line in string.split('\n'):
        line = line.strip()
        if line == "":
            continue
        # remove comments, since they may have forward-slashes
        # set ignore_comments to True is you want to keep them.
        if line.startswith('!'):
            continue  
        if ignore_comments and '!' in line:
            line = line[:line.index('!')]

        filtered_lines.append(line)

    group_blocks = re.findall(group_re, "\n".join(filtered_lines))

    for i, group_block in enumerate(group_blocks):
        group_lines = group_block.split('\n')
        group_name = group_lines.pop(0).strip()
        # check for comments
        if "!" in group_name:
            i = group_name.index("!")
            group_name = group_name[:i].strip()
            group_help = group_name[i+1:].strip()

        # some lines are continuation of previous lines: filter
        joined_lines = []
        for line in group_lines:
            line = line.strip()
            if '=' in line:
                joined_lines.append(line)
            else:
                # continuation of previous line
                joined_lines[-1] += line
        group_lines = joined_lines

        for line in group_lines:
            name, value, comment = _parse_line(line)

            param = ParamNml(name, value, help=comment, group=group_name)

            # group[variable_name] = parsed_value
            params.append(param)

        # groups[group_name] = group
    return params

def _parse_line(line):
    "parse a line within a block"
    # commas at the end of lines seem to be optional
    comment = ""
    if '!' in line:
        sep = line.index("!")
        comment = line[sep+1:].strip()
        line = line[:sep].strip()

    if line.endswith(','):
        line = line[:-1]

    k, v = line.split('=')
    name = k.strip()
    value = _parse_value(v.strip())
    return name, value, comment

def _parse_value(variable_value):
    """
    Tries to parse a single value, raises an exception if no single value is matched
    """
    try:
        parsed_value = int(variable_value)
    except ValueError:
        try:
            parsed_value = float(variable_value)
        except ValueError:
            if variable_value.lower() in ['.true.', 't', 'true']:
                # boolean
                parsed_value = True
            elif variable_value.lower() in ['.false.', 'f', 'false']:
                parsed_value = False
            elif variable_value.startswith("'") \
                and variable_value.endswith("'") \
                and variable_value.count("'") == 2 \
            or variable_value.startswith('"') \
                and variable_value.endswith('"') \
                and variable_value.count('"') == 2:
                parsed_value = variable_value[1:-1]
            elif variable_value.startswith("/") and variable_value.endswith("/"):
                # array /3,4,5/
                parsed_value = _parse_array(variable_value[1:-1].split(','))
            elif "," in variable_value:
                # array 3, 4, 5
                parsed_value = _parse_array(variable_value.split(','))
            elif '*' in variable_value:
                # 3*4  means [4, 4, 4, 4] ==> this is handled in _parse_array
                parsed_value = _parse_array([variable_value])
            elif len(variable_value.split()) > 1:
                # array 3 4 5
                parsed_value = _parse_array(variable_value.split())
            else:
                print("Parsing ERROR: >>>{}<<<".format(variable_value))
                raise ValueError(variable_value)
    return parsed_value

def _parse_array(values):
    """ parse a list of (string) values representing a fortran array
    and return a python list
    """
    assert type(values) is list
    parsed_value = []
    for v in values:
        if '*' in v:
            # 3* "a" === "a", "a", "a"
            mult, val = v.split('*')
            parsed_value.extend(int(mult) * [ _parse_value(val.strip())  ])
        else:
            parsed_value.append(_parse_value(v))
    return parsed_value

def format_nml(params):
    """ format a flat parameter list to be written in the namelist
    """
    lines = []
    for group_name, group_params in groupby(params, lambda x: x.group):
        if group_name == "":
            print(list(group_params))
            raise ValueError("Group not defined. Cannot write to namelist.")
        lines.append("&{}".format(group_name))
        for param in group_params:
            if isinstance(param.value, list):
                nmstr = "{:15}".format(param.name)
                line = " {} = {}".format(nmstr, " ".join([_format_value(v) for v in param.value]))
            else:
                nmstr = "{:15}".format(param.name)
                line = " {} = {}".format(nmstr, _format_value(param.value))
            line = "{:30}".format(line)
            if param.help:
                line += ' ! '+param.help
            lines.append(line)
        lines.append("/")
    return "\n".join(lines) + "\n"

def _format_value(value):
    """ Format a value into fortran's namelist format (return a string)
    """
    if isinstance(value, bool):
        return value and '.true.' or '.false.'
    else:
        return "{}".format(repr(value))

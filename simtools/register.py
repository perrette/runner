import argparse
import warnings

jobs = []
filetypes = {}

# model custom
model = { 
    'command': None,
    'setup': None,
    'getvar': None,
    'loads': None,
    'dumps': None,
}

_defaults = {}

def register_job(name, parser, postproc, help=None):
    job = argparse.Namespace(name=name, 
                             parser=parser, 
                             postproc=postproc,
                             help=help)
    _defaults[name] = {}
    jobs.append(job)

def set_defaults(*cmds, **kwargs):
    """set default arguments for a subcommand (ArgumentParser.set_defaults)

    *cmds : command for which the defaults apply 
        (by default, everything already registered)
    **kwargs : key-word arguments
    """
    if not cmds:
        cmds = _defaults.keys()
    for cmd in cmds:
        if cmd in _defaults:
            _defaults[cmd].update(kwargs)


def register_filetype(name, filetype):
    if name in filetypes:
        warnings.warn("filetype name already exists: "+repr(name))
    filetypes[name] = filetype


def _check_free(key):
    if model[key]:
        warnings.warn(key + ' was already registered : overwrite')
        return False
    else:
        return True


def define_model(command=None, setup=None, getvar=None, dumps=None, loads=None, defaults=None):
    """
    - setup : callable ( rundir, executable, *args, **params )
        prepare run directory (e.g. write param file)
    - command : callable ( rundir, executable, *args, **params ) --> list of args
        make run command given output directory and parameters
    - getvar : callable ( name, rundir, executable, *args, **params ) --> state variable (scalar)
    - loads : callable ( file content ) --> params dict {name:value}
    - dumps : callable ( params dict ) --> file content (string)
    **kwargs : will be used to set parser defaults with job run (e.g. executable etc)
        NOTE: this will only affects already registered job, so for the defaults
        to affect all commands, make sure to import job module
    
    >>> import simtools.job   # job is defined and all commands registered
    >>> from simtools.register import define_model
    >>> define_model(executable=...)
    """
    if setup or _check_free("setup"):
        model["setup"] = setup

    if command or _check_free("command"):
        model["command"] = command

    if getvar or _check_free("getvar"):
        model["getvar"] = getvar

    if dumps or _check_free("dumps"):
        model["dumps"] = dumps

    if loads or _check_free("loads"):
        model["loads"] = loads

    if defaults:
        set_defaults(defaults)


# decorator:

class Model(object):
    def __init__(self, cmd=None):
        self.cmd = cmd

    def __call__(self, func):
        define_model(**{self.cmd:self.cmd})
        return func

# to access as @define.command, @define.setup
for cmd in ["command", "setup", "getvar", "dumps", "loads"]:
    setattr(Model, cmd, property(lambda self: Model(cmd)))

define = Model()

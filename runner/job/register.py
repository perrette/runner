import argparse
import warnings

jobs = []
filetypes = {}

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

def register_filetype(name, filetype, *ext):
    if name in filetypes:
        warnings.warn("filetype name already exists: "+repr(name))
    filetypes[name] = filetype, ext

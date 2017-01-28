import argparse
import warnings

jobs = []
filetypes = {}

# model custom
model = { 
    'command': None,
    'setup': None,
    'getvar': None,
    'filetype': None,
}

defaults = {}

def register_job(name, parser, postproc, help=None):
    job = argparse.Namespace(name=name, 
                             parser=parser, 
                             postproc=postproc,
                             help=help)
    defaults[name] = {}
    jobs.append(job)

def register_filetype(name, filetype):
    if name in filetypes:
        warnings.warn("filetype name already exists: "+repr(name))
    #if not hasattr(filetype, 'dumps'):
    #    raise TypeError("file type must have a `dumps` method")
    filetypes[name] = filetype


def register_model(command=None, setup=None, getvar=None, filetype=None, **kwargs):
    """
    - setup : callable ( rundir, executable, *args, **params )
        prepare run directory (e.g. write param file)
    - command : callable ( rundir, executable, *args, **params ) --> list of args
        make run command given output directory and parameters
    - getvar : callable ( name, rundir ) --> state variable (scalar)
    - filetype : `dumps` and `loads` methods, operates on `dict` parameters
    **kwargs : will be used to set parser defaults with job run (e.g. executable etc)
        NOTE: this will only affects already registered job, so for the defaults
        (but not the filetype) to affect all commands, make sure to import job module:
    
    >>> import simtools.job   # job is defined and all commands registered
    >>> from simtools.register import register_model
    >>> register_model(..., filetype=...)
    """
    if command:
        model["command"] = command
    if setup:
        model["setup"] = setup
    if getvar:
        model["getvar"] = getvar

    if filetype:
        from simtools.filetype import FileTypeWrapper
        register_filetype("custom", FileTypeWrapper(filetype))
        model["filetype"] = filetypes["custom"]
        assert 'file_type' not in kwargs or file_type == "custom"
        kwargs["file_type"] = "custom"

    if kwargs:
        for cmd in defaults:
            defaults[cmd].update(**kwargs)

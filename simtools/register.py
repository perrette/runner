import argparse

jobs = []
filetypes = {}

# model custom
model = { 
    'command': None,
    'setup': None,
    'getvar': None,
}

def register_job(name, parser, postproc, help=None):
    job = argparse.Namespace(name=name, 
                             parser=parser, 
                             postproc=postproc,
                             help=help)
    jobs.append(job)

def register_filetype(name, filetype):
    if name in filetypes:
        raise ValueError("filetype name already exists: "+repr(name))
    if not hasattr(filetype, 'dumps'):
        raise TypeError("file type must have a `dumps` method")
    filetypes[name] = filetype


def register_model(command=None, setup=None, getvar=None):
    """
    command : callable ( rundir, executable, *args, **params ) --> list of args
    setup : callable ( rundir, executable, *args, **params )
    getvar : callable ( name, rundir ) --> state variable (scalar)
    """
    if command:
        model["command"] = command
    if setup:
        model["setup"] = setup
    if getvar:
        model["getvar"] = getvar

import argparse

jobs = []
filetypes = {}

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

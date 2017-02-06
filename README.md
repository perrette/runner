# job

The interface between job and the model is dealt with with appropriate command-line
arguments, once and for all thanks to save/load system of options.
Nevertheless it is possible to defined custom functions in python, for greater control
or clarity for what is being done. 

Examples:

>>> from runner.register import define

>>> @define.command
>>> def make_command(rundir, exe, *args, **params):
...     cmd = [exe, "--out", rundir]
...     for k in params:
...           


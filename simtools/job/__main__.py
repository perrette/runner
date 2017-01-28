#!/usr/bin/env python
"""Jobs for numerical experiments
"""
import sys
from importlib import import_module
import argparse
import warnings
import json
from simtools import __version__
from simtools import register

# job config I/O
# ==============
def _parser_defaults(parser):
    " parser default values "
    return {a.dest: a.default for a in parser._actions}


def _modified(kw, defaults):
    """return key-words that are different from default parser values
    """
    return {k:kw[k] for k in kw if k in defaults and kw[k] != defaults[k]}

def _filter(kw, after, diff=False, include_none=True):
    if diff:
        filtered = _modified(kw, after)
    else:
        filtered = {k:kw[k] for k in kw if k in after}
    if not include_none:
        filtered = {k:filtered[k] for k in filtered if filtered[k] is not None}
    return filtered

def json_config(cfg, defaults=None, diff=False, name=None):
    import datetime
    js = {
        'defaults': _filter(cfg, defaults, diff) if defaults is not None else cfg,
        'version':__version__,
        'date':str(datetime.date.today()),
        'name':name,  # just as metadata
    }
    return json.dumps(js, indent=2, sort_keys=True, default=lambda x: str(x))

def write_config(cfg, file, defaults=None, diff=False, name=None):
    string = json_config(cfg, defaults, diff, name)
    with open(file, 'w') as f:
        f.write(string)



# pull main job together
# ======================
def main(argv=None):

    # prepare parser
    job = argparse.ArgumentParser(parents=[], description=__doc__, 
                                  formatter_class=argparse.RawTextHelpFormatter)

    job.add_argument('-v','--version', action='version', version=__version__)
    job.add_argument('-m','--module', nargs='+',
                     help='load python module(s) that contain custom file type or model definitions (see simtools.register)')
    job.add_argument('-c','--config-file', 
                        help='load defaults from configuration file')
    x = job.add_mutually_exclusive_group()
    x.add_argument('-s','--saveas', action="store_true", 
                   help='save selected defaults to config file and exit')
    x.add_argument('-u', '--update-config', action="store_true", 
                        help='-uc FILE is an alias for -c FILE -s FILE')
    job.add_argument('--show', action="store_true", help='show config and exit')
    
    top = argparse.ArgumentParser(parents=[job], conflict_handler='resolve')

    # add subcommands
    subp = job.add_subparsers(dest='cmd')
    postprocs = {}
    parsers = {}

    for j in register.jobs:
        subp.add_parser(j.name, parents=[j.parser], help=j.help)
        parsers[j.name] = j.parser
        postprocs[j.name] = j.postproc

    if argv is None:
        argv = sys.argv[1:]

    # parse arguments and select sub-parser
    o = job.parse_args(argv)
    parser = parsers[o.cmd]
    func = postprocs[o.cmd]

    # now make sure subparse does not interfer
    i = argv.index(o.cmd)
    topargs = argv[:i]
    cmdargs = argv[i+1:]
    o = top.parse_args(topargs)  # no subcommands

    if o.module:
        for m in o.module:
            import_module(m)

    # parse again with updated defaults
    defaults = register._defaults[o.cmd].copy()  # --module

    # read config file?
    if o.config_file:

        js = json.load(open(o.config_file))
        if js["name"] != o.cmd:
            warnings.warn("config file created from another command")

        defaults.update(js["defaults"])
        
    parser.set_defaults(**defaults)

    # now subparser 
    cmdo = parser.parse_args(cmdargs)

    if o.update_config:
        o.saveas = o.config_file

    # save to file?
    if o.saveas or o.show:
        #saveable = _filter(o.__dict__, global_defaults, diff=False, include_none=False)
        saveable = _filter(cmdo.__dict__, _parser_defaults(parser), diff=False, include_none=False)
        string = json_config(saveable, name=o.cmdo)
        if o.saveas:
            with open(o.saveas, 'w') as f:
                f.write(string)
        if o.show:
            print(string)
        return

    return func(cmdo)

if __name__ == '__main__':
    main()

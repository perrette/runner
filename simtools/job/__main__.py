#!/usr/bin/env python
"""Jobs for numerical experiments
"""
from __future__ import absolute_import
import sys, os
from importlib import import_module
import argparse
import warnings
from simtools import register
from .config import json_config, load_config, __version__




# pull main job together
# ======================
def main(argv=None):

    # prepare parser
    job = argparse.ArgumentParser(parents=[], description=__doc__, 
                                  formatter_class=argparse.RawTextHelpFormatter)

    job.add_argument('-v','--version', action='version', version=__version__)
    job.add_argument('-c','--config-file', 
                        help='load defaults from configuration file')
    x = job.add_mutually_exclusive_group()
    x.add_argument('-s','--saveas', 
                   help='save selected defaults to config file and exit')
    x.add_argument('-u', '--update-config', action="store_true", 
                        help='-uc FILE is an alias for -c FILE -s FILE')
    job.add_argument('--show', action="store_true", help='show config and exit')
    
    top = argparse.ArgumentParser(parents=[job], conflict_handler='resolve')
    tops = top.add_subparsers(dest='cmd') # just for the command

    # add subcommands
    subp = job.add_subparsers(dest='cmd')
    postprocs = {}
    parsers = {}

    for j in register.jobs:
        subp.add_parser(j.name, parents=[j.parser], help=j.help)
        tops.add_parser(j.name, help=j.help)
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
    topargs = argv[:i+1] # include subcommand
    cmdargs = argv[i+1:]
    o = top.parse_args(topargs)  # no subcommands

    # parse again with updated defaults
    defaults = register._defaults[o.cmd].copy()  # --module

    # read config file?
    if o.config_file:
        js = load_config(o.config_file, parser)
        defaults.update(js)
        
    parser.set_defaults(**defaults)

    # now subparser 
    cmdo = parser.parse_args(cmdargs)

    if o.update_config:
        o.saveas = o.config_file

    # save to file?
    if o.saveas or o.show:
        string = json_config(cmdo.__dict__, parser)
        if o.saveas:
            with open(o.saveas, 'w') as f:
                f.write(string)
        if o.show:
            print(string)
        return

    return func(cmdo)

if __name__ == '__main__':
    main()

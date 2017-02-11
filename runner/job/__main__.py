#!/usr/bin/env python
"""Jobs for numerical experiments
"""
from __future__ import absolute_import
import sys, os
from importlib import import_module
import argparse
import logging
from runner import register
from .config import json_config, load_config, __version__




# pull main job together
# ======================
def main(argv=None):

    # prepare parser
    job = argparse.ArgumentParser('job', parents=[], description=__doc__, 
                                  formatter_class=argparse.RawDescriptionHelpFormatter)

    job.add_argument('-v','--version', action='version', version=__version__)
    job.add_argument('-c','--config-file', 
                        help='load defaults from configuration file')
    x = job.add_mutually_exclusive_group()
    x.add_argument('-s','--saveas', 
                   help='save selected defaults to config file and exit')
    x.add_argument('-u', '--update-config', action="store_true", 
                        help='-uc FILE is an alias for -c FILE -s FILE')
    job.add_argument('--show', action="store_true", help='show config and exit')
    job.add_argument('--full', action='store_false', dest='diff',
                   help='save/show full config, not only differences from default')
    job.add_argument('--debug', action="store_true", help='print full traceback')
    
    top = argparse.ArgumentParser(parents=[job], add_help=False)
    tops = top.add_subparsers(dest='cmd') # just for the command

    # add subcommands
    subp = job.add_subparsers(dest='cmd')
    postprocs = {}
    parsers = {}

    for j in register.jobs:
        subp.add_parser(j.name, 
                        parents=[j.parser], 
                        add_help=False, 
                        description=j.parser.description, 
                        epilog=j.parser.epilog, 
                        help=j.help,
                        formatter_class=j.parser.formatter_class)
        tops.add_parser(j.name, help=j.help, add_help=False)
        parsers[j.name] = j.parser
        postprocs[j.name] = j.postproc

    if argv is None:
        argv = sys.argv[1:]

    # parse arguments and select sub-parser
    o = job.parse_args(argv)
    parser = parsers[o.cmd]
    func = postprocs[o.cmd]

    if o.debug:
        logging.basicConfig(level=logging.DEBUG)

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
        string = json_config(cmdo.__dict__, parser, diff=o.diff)
        if o.saveas:
            with open(o.saveas, 'w') as f:
                f.write(string)
        if o.show:
            print(string)
        return

    try:
        func(cmdo)
    except Exception as error:
        if o.debug:
            raise
        else:
            print("ERROR: "+str(error))
            print("ERROR: use --debug to print full traceback")
            job.exit(1)

if __name__ == '__main__':
    main()

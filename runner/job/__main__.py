#!/usr/bin/env python
"""Jobs for numerical experiments
"""
from __future__ import absolute_import
import sys, os
from importlib import import_module
import argparse
import logging
from runner.job import register
from runner import __version__


# pull main job together
# ======================
def main(argv=None):

    # prepare parser
    job = argparse.ArgumentParser('job', parents=[], description=__doc__, 
                                  formatter_class=argparse.RawDescriptionHelpFormatter)

    job.add_argument('-v','--version', action='version', version=__version__)
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

    # now subparser 
    cmdo = parser.parse_args(cmdargs)

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

#!/usr/bin/env python
"""Jobs for numerical experiments
"""
from __future__ import absolute_import
import sys, os
from importlib import import_module
import argparse
import logging
from runner import __version__
from runner.job.tools import jobs

# import module to register job
from runner.job import stats, setup, run, analysis, iis


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

    for name,j in jobs.items():
        subp.add_parser(name, 
                        parents=[j.parser], 
                        add_help=False, 
                        description=j.parser.description, 
                        epilog=j.parser.epilog, 
                        help=j.help,
                        formatter_class=j.parser.formatter_class)
        tops.add_parser(name, help=j.help, add_help=False)

    if argv is None:
        argv = sys.argv[1:]

    # parse arguments and select sub-parser
    o = job.parse_args(argv)
    j = jobs[o.cmd]

    if o.debug:
        logging.basicConfig(level=logging.DEBUG)

    # now make sure subparse does not interfer
    i = argv.index(o.cmd)
    topargs = argv[:i+1] # include subcommand
    cmdargs = argv[i+1:]
    o = top.parse_args(topargs)  # no subcommands

    # now subparser 
    if j.init:
        j.init(j.parser, cmdargs)
    cmdo = j.parser.parse_args(cmdargs)

    try:
        j.run(cmdo)
    except Exception as error:
        if o.debug:
            raise
        else:
            print("ERROR: "+str(error))
            print("ERROR: use --debug to print full traceback")
            job.exit(1)


if __name__ == '__main__':
    main()

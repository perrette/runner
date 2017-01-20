"""Decentralize Job scripts
"""
from __future__ import print_function
import argparse
import inspect
import sys

class CustomParser(argparse.ArgumentParser):
    """Program with command-line arguments.
    """
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self._postprocessors = []

        # also copy postprocessors from parents
        for parent in kwargs.pop('parents', []):
            if hasattr(parent, '_postprocessors'):
                self._postprocessors.extend(parent._postprocessors)


    #TODO: think about simplifying the postprocessor trigger (which takes a namespace
    # as argument) and the function <--> namespace mapping.
    # ==> make a function decorator **kwargs --> namespace
    # and a method `add_arguments(parser...)` 
    def add_postprocessor(self, func, args=None, optargs=(), 
                          varargs=(), inspect=False, mapping=None, dest=None, add_arguments=False):
        """Add a group of parser 
        
        Parameters
        ----------
        func : callable
        args : [str], select namespace arguments to be passed to func
            (if None, all namespace arguments are passed)
        optargs : optional arguments, not required to be in the namespace
            (the default behaviour is to raise an error if the argument was not found)
        varargs : [str], namespace arguments to be passed as variable arguments
            No mapping is applied on these arguments, and they must be present in the namespace.
        inspect : bool, false by default
            if True, use inspect function to guess args and optargs
        mapping : {kwargs : destarg}, mapping between function key-word arguments 
            to command arguments (the `dest` keyword in `add_argument`). 
        dest : str or [None], destination of `func` return value in the result 
            namespace. By default no result is stored.
        add_arguments : bool, if True, also add postprocessor arguments to the parser

        Examples
        --------
        >>> def func(required, integer=1, real=1., string='a', ofcourse=True, really=False, none=None):
        ...    return locals()
        >>> 
        >>> p = CustomParser('test')
        >>> p.add_postprocessor(func, inspect=True, add_arguments=True, dest='result')
        >>> p.print_help()
        """
        if mapping is None:
            mapping = {}

        if inspect:
            from inspect import getargspec
            spec = getargspec(func)
            args = spec.args
            optargs = spec.args[-len(spec.defaults):]
            defaults = {arg:val for  arg,val in zip(optargs, spec.defaults)}
        else:
            defaults = {}

        args = [a for a in args if a not in optargs] # make args and optargs disjoint

        self._postprocessors.append((func, args, optargs, varargs, mapping, dest))

        # add arguments to parser ?
        if add_arguments:
            for arg in args + optargs:
                if arg in optargs:
                    self._add_optional_argument(arg, default=defaults.pop(arg, None))
                else:
                    self._add_optional_argument(arg, required=True)


    def _add_optional_argument(self, dest, default=None, aliases=(), required=False, **kwargs):
        """Add an optional argument to parser based on its dest name and default value

        (called by add_optional_argument)

        dest : same as `add_argument`
            The flag name `--ARG-NAME` or `--no-ARG-NAME` is built from `dest`.
        default : default value
            `type` and help is derived from default.
            If boolean, `store_true` or `store_false` will be determined, and 
            name updated.
        required : same as `add_argument`
        aliases : [str], additional aliases such as `--OTHER-NAME` or `-X`
        **kwargs keyword arguments are passed to `add_argument`
        """
        name = dest.replace('_','-')

        opt = {}

        if type(default) is bool:
            if default is False:
                opt["action"] = "store_true"
            else:
                opt["action"] = "store_false"
                name = 'no-'+name
        elif default is not None:
            opt["type"] = type(default)

        # determine help
        if default is not None and not required:
            if default is False:
                opt["help"] = '[default: False]'
            elif default is True:
                opt["help"] = '[default: True]'
            else:
                opt["help"] = '[default: %(default)s]'

        # update with user-specified
        opt.update(kwargs)

        return self.add_argument('--'+name, *aliases, dest=dest, required=required, default=default, **opt)


    def postprocess(self, namespace, results=None):
        """Postprocess namespace

        namespace : Namespace instance
            returned by parse_args

        results : Namespace instance, optional
            if results is not provided, the original namespace
            will be populated accordingly to `dest` keyword
            in add_postprocessor
        """
        if results is None:
            results = namespace

        for func, args, optargs, varargs, mapping, dest in self._postprocessors:

            # determine arguments to extract from namespace
            if args is None:
                args = namespace.__dict__.keys()

            # extract arguments from namespace
            kwargs = {arg: getattr(namespace, arg) for arg in args if args not in optargs}
            kwargs_opt = {arg: getattr(namespace, arg) for arg in optargs if hasattr(namespace, arg)}
            kwargs.update(kwargs_opt)

            # mapping toward function name
            mapping = mapping.copy()

            varvals = [getattr(namespace, arg) for k in varargs]

            res = func(*varvals, **kwargs)

            if dest is not None:
                setattr(results, dest, res)

        return results


    def main(self, argv=None, namespace=None):
        """Callable as main function: parse arguments 
        and return custom namespace.
        """
        namespace = self.parse_args(argv, namespace)
        return self.postprocess(namespace)


    def __call__(self, func):
        """To use as a decorator
        """
        # if no postprocessor is present, build a default postproc from func
        if not self._postprocessors:
            self.add_postprocessor(func, inspect=True, add_arguments=True)

        return self.main

def parser(func):
    """Decorator to make the function `main`
    """
    parser = CustomParser(description=func.__doc__,
                          formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_postprocessor(func, inspect=True, add_arguments=True)
    return parser.main



class Job(object):
    """Multiple main functions organized as sub-commands.
    """
    def __init__(self, dest="cmd", **kwargs):

        self.parser = argparse.ArgumentParser(**kwargs)
        self.subparsers = self.parser.add_subparsers(dest=dest)
        self.commands = {}
        self.dest = dest

    def add_command(self, name, command, **kwargs):
        """Register a command (`command`)
        """
        assert callable(command)
        subparser = self.subparsers.add_parser(name, **kwargs)
        self.commands[name] = command

    def main(self, argv=None):
        """Exectute program by calling the command
        """
        if argv is None:
            argv = sys.argv[1:]

        if '--debug' in argv:
            debug = True
            argv.remove('--debug') 
        else:
            debug = False

        # if subcommand, just call it:
        if len(argv) > 0:
            if argv[0] in self.commands:
                program = self.commands[argv[0]]
                sys.argv[0] = " ".join(sys.argv[:2]) # for help message
                try:
                    return program(argv[1:])
                except Exception as error:
                    if debug:
                        raise
                    print("ERROR:",error.message)
                    print("(use '--debug' for full traceback)")
                    sys.exit(1)


        # error message and help:
        self.parser.parse_args(argv)

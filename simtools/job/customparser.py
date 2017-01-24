from __future__ import print_function
import argparse
import inspect
import sys
import json


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
            (if None and no mapping, all namespace arguments are passed)
        optargs : optional arguments, not required to be in the namespace
            (the default behaviour is to raise an error if the argument was not found)
        varargs : [str], namespace arguments to be passed as variable arguments
            No mapping is applied on these arguments, and they must be present in the namespace.
        inspect : bool, false by default
            if True, use inspect function to guess args and optargs
        mapping : {kwargs : destarg}, mapping between function key-word arguments 
            to command arguments (the `dest` keyword in `add_argument`) --> added to args
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

            # inspect retrieves function argument --> apply mapping towards namespace
            if mapping:
                args = [mapping.copy().pop(a, a) for a in args]
                optargs = [mapping.copy().pop(a, a) for a in optargs]
                defaults = {mapping.copy().pop(a, a):defaults[a] for a in defaults}
        else:
            defaults = {}

        # add arguments from mapping
        for a in mapping:
            if mapping[a] not in (args or []) and mapping[a] not in (optargs or []):
                args = (args or []) + [mapping[a]]

        if args is not None:
            args = [a for a in args if a not in optargs] # make args and optargs disjoint

        self._postprocessors.append((func, args, optargs, varargs, mapping, dest))

        # add arguments to parser ?
        if add_arguments:
            for arg in args + optargs:
                try:
                    help = grepdoc(func.__doc__, arg)
                except:
                    help = None
                if arg in optargs:
                    add_optional_argument(self, arg, default=defaults.pop(arg, None), help=help)
                else:
                    add_optional_argument(self, arg, required=True, help=help)


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
            for k in mapping:
                kwargs[k] = kwargs.pop(mapping[k])

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



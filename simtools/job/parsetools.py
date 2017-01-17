"""Decentralize Job scripts
"""
import argparse
import inspect

class ObjectParser(object):
    """Bundle argument definition and postprocessing in the same class.

    ObjectParser helps to define and process a group of arguments in an integrated
    manner, such as to initialize a class. 
    This is useful to deal with a large number of arguments and their 
    re-use in various subcommands.
    """
    def __init__(self, **kwargs):
        """Parameterize the parser --> anything here will be available in its methods
        """
        self._description = (self.__doc__ or "").strip()
        self.__dict__.update(kwargs)

    def add_arguments(self, parser):
        """Add arguments to a argparse.ArgumentParser instance
        """
        raise NotImplementedError()

    def parse_args(self, argv=None, formatter_class=argparse.RawDescriptionHelpFormatter, **kwargs):
        """Parse command-line arguments and return argparse.Namespace
        """
        description = kwargs.pop("description", self._description)
        parser = argparse.ArgumentParser(description=description,
                                         formatter_class=formatter_class, **kwargs)
        self.add_arguments(parser)
        namespace = parser.parse_args(argv)
        return namespace

    def postprocess(self, namespace):
        """Return desired object based on argparse.Namespace instance
        """
        raise NotImplementedError()

    def __call__(self, argv=None):
        """Perform action based on command-line args
        """
        namespace = self.parse_args(argv, *args, **kwargs)
        return self.postprocess(namespace)


class Program(object):
    """Program with command-line arguments.
    """
    def __init__(self, description=(__doc__ or "").strip(), 
                 formatter_class=argparse.RawDescriptionHelpFormatter, **kwargs):

        self.parser = argparse.ArgumentParser(description=description,
                formatter_class=formatter_class, **kwargs)


    def add_object_arguments(self, objectparser, *args, **kwargs):
        """Add a group of parser 
        
        objectparser = ObjectParser instance
        *args and **kwargs are passed to ArgumentParser.add_argument_group
        """
        grp = self.parser.add_argument_group(*args, **kwargs)
        objectparser.add_argument(grp)


    def main(self, namespace):
        """Main function taking argparse.Namespace instance as argument.
        """
        raise NotImplementedError()


    def __call__(self, argv=None):
        namespace = self.parser.parse_args(argv)
        return self.main(namespace)



class Job(object):
    """Multiple programs organized as sub-commands.
    """
    def __init__(self, dest="cmd", description=(__doc__ or "").strip(), 
                 formatter_class=argparse.RawDescriptionHelpFormatter, **kwargs):

        self.parser = argparse.ArgumentParser(description=description,
                formatter_class=formatter_class, **kwargs)

        self.subparsers = self.parser.add_subparsers(dest=dest)
        self.commands = {}
        self.dest = dest

    def add_command(self, name, program, **kwargs):
        """Register a command
        """
        desc = kwargs.pop("description", program.__doc__)
        subparser = self.subparsers.add_parser(name, description=desc, **kwargs)
        program.add_arguments(subparser)
        self.commands[name] = program

    def __call__(self, argv=None):
        """Exectute program by calling the command
        """
        namespace = self.parser.parse_args(argv)
        program = self.commands[namespace.dest]
        return program.main(namespace)


#############################################################
#
#
#
# Result of intense procrastination, but in the end powerful :

class FunWrapper(ObjectParser):
    """Automatically create a parser for a function or class

    This class is pretty rich and lets you a lot of flexibility about 
    how to define the argparse wrapper.

    The basic usage is to provide it with a callable with its keyword
    arguments, and optionally with a variable list of key-word arguments
    used to specify the order in which they shall appear.

    Take a look at the various classes to see how the default behaviour can be 
    modified (e.g. argument mapping, aliases, require, hide ...). 
    Or on the opposite end, use the `inspect` class method to automatically
    determine the default arguments.
    """
    def __init__(self, fun, *args, **kwargs):
        """
        fun : callable
        *args : may be provided to specify order
        **kwargs : default key-word arguments for fun.
        """
        self.fun = fun
        self._arguments = args or sorted(kwargs.keys())
        self._defaults = kwargs
        self._description = (fun.__doc__ or "").strip()
        self._required = []
        self._hidden = []
        self._aliases = {}
        self._added_arguments = []
        self._mapping = {}
        self._custom_update = {} 

    def require(self, k):
        """Require the argument from the command-line (no default)
        """
        self._required.append(k)

    def hide(self, k):
        """Hide argument from command-line usage and help
        """
        self._hidden.append(k)

    def alias(self, k, aliases=None):
        """Add aliases to the parser

        aliases : list of aliases, optional
            by default just use the first letter
        """
        self._aliases[k] = aliases or ('-'+k[0],)

    def map(self, **kwargs):
        """Provide a mapping for the arguments
        """
        self._mapping.update(kwargs)


    def add_argument(self, *args, **kwargs):
        """Add single argument to parser (passed to ArgumentParser.add_argument)
        """
        self._added_arguments.append( (args, kwargs) )

    def update_argument(self, k, *aliases, **kwargs):
        """Update specification compared to default
        """
        self._custom_update[k] = aliases, kwargs

    def add_arguments(self, parser):
        """Populate parser
        """
        for k in self._arguments:

            name = k.replace('_','-')
            args = ('--'+name,) + self._aliases.copy().pop(k, ())
            kwargs = {}

            default = self._defaults[k]
            kwargs["default"] = default

            if type(default) is bool:
                if default is False:
                    kwargs["action"] = "store_true"
                else:
                    kwargs["action"] = "store_false"
                    args = ('--no-'+name,) + args[1:]
            else:
                kwargs["type"] = type(default)

            if k in self._required:
                kwargs["required"] = True

            if k in self._hidden:
                kwargs["help"] = argparse.SUPPRESS
            elif default is False:
                kwargs["help"] = 'set [unset by default]'
            elif default is True:
                kwargs["help"] = 'unset [set by default]'
            elif k not in self._required:
                kwargs["help"] = '[default: %(default)s]'


            kwargs["dest"] = k
            
            # custom update: replace args (aliases), update kwargs
            args, update = self._custom_update.copy().pop(k,(args, {}))
            kwargs.update(update)

            parser.add_argument(*args, **kwargs)

        # Add custom arguments as indicated by user
        for args, kwargs in self._added_arguments:
            parser.add_argument(*args, **kwargs)

    def postprocess(self, namespace):
        kwargs = namespace.__dict__.copy()
        for k in self._mapping:
            kwargs[k] = kwargs.pop(self._mapping[k])
        return self.fun(**kwargs)


    @staticmethod
    def _inspect_order(fun):
        # remove self or cls as first argument
        args = inspect.getargspec(fun).args
        if not inspect.isfunction(fun):
            args = args[1:]
        return args

    @staticmethod
    def _inspect_defaults(fun):
        spec = inspect.getargspec(fun)
        default_list = spec.defaults or ()
        defaults = {spec.args[::-1][i] : val 
                    for i,val in enumerate(default_list[::-1])}
        return defaults

    @classmethod
    def inspect(cls, fun, **kwargs):
        """initialize Wrapper from inspect module

        func : function or callable
        **kwargs : update or specify default arguments for func
        """
        args = cls._inspect_order(fun)
        defaults = cls._inspect_defaults(fun)
        defaults.update(kwargs)
        undefined = [a for a in args if a not in defaults]
        if undefined:
            raise ValueError("undefined arguments: "+",".join(undefined))
        return cls(fun, *args, **defaults)

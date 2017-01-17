"""Decentralize Job scripts !
"""
import argparse

class Command(object):
    """One job to be executed as subcommand or main command
    """
    def __init__(self, parser):
        raise NotImplementedError("parser definition")

    def __call__(self, namespace):
        raise NotImplementedError("main job")


    @classmethod
    def main(cls, argv=None, formatter_class=argparse.RawDescriptionHelpFormatter, **kwargs):
        """To execute the command as a main job
        """
        description = kwargs.pop("description", cls.__doc__)
        parser = argparse.ArgumentParser(description=description,
                                              formatter_class=formatter_class, **kwargs)
        cmd = cls(parser)
        args = parser.parse_args(argv)
        return cmd(args)


class Job(object):
    """Put a main function together with several jobs as subcommands
    """
    def __init__(self, dest="cmd", description=__doc__, 
                 formatter_class=argparse.RawDescriptionHelpFormatter, **kwargs):

        self.parser = ParamsParser(description=description,
                formatter_class=formatter_class, **kwargs)

        self.subparsers = self.parser.add_subparsers(dest=dest)
        self.commands = {}
        self.dest = dest

    def add_command(self, name, Cmd, **kwargs):
        """Register a command
        """
        desc = kwargs.pop("description", Cmd.__doc__)
        subp = self.subparsers.add_parser(name, description=desc)
        cmd = Cmd(subp)  # define arguments and return callable
        self.commands[name] = cmd  # save for future exec

    def main(self, *args, **kwargs):
        """Exectute program by calling the command
        """
        args = self.parser.parse_args(*args, **kwargs)
        cmd = self.commands[args.dest]
        return cmd(args)

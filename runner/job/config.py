import argparse
from collections import OrderedDict as odict
import datetime
import json
from runner import __version__

class ParserIO(object):

    def __init__(self, parser, dump_filter=None, load_filter=None, get=None):
        """
        * parser : argparse.ArgumentParser instance
        """
        self.parser = parser
        self._dump_filter = dump_filter or self._filter
        self._load_filter = load_filter or self._filter
        self.get = get

    def _names(self):
        for a in self.parser._actions: 
            yield a.dest

    def _filter(self, dict_):
        return odict([(k,v) for k,v in dict_.items() if k in self._names()])

    def _get_defaults(self):
        return {name:self.parser.get_default(name) for name in self._names()}

    def namespace(self, **kwargs):
        opt = self._get_defaults()
        opt.update(self._filter(kwargs))
        return argparse.Namespace(**opt)

    def dumps(self, namespace, name=None, indent=2, **kwargs):
        js = {
            'defaults': self._dump_filter(vars(namespace)),
            'version':__version__,
            'date':str(datetime.date.today()),
            'name':name,  # just as metadata
        }
        return json.dumps(js, indent=indent, **kwargs)

    def loads(self, string, update={}):
        js = json.loads(string)
        js = self._load_filter(js)
        js.update(update)
        return self.namespace(**js)

    def dump(self, namespace, file, **kwargs):
        file.write(self.dumps(namespace, **kwargs))

    def load(self, file, update={}):
        return self.loads(file.read(), update)


    def join(self, other, **kwargs):
        " for I/O only, forget about get "
        parser = argparse.ArgumentParser(add_help=False, 
                                         parents=[self.parser, other.parser], **kwargs)
        return ParserIO(parser, 
                        lambda x: other._dump_filter(self._dump_filter(x)),
                        lambda x: other._load_filter(self._load_filter(x)),
                        get = self.get or other.get)


jobs = odict()

class Job(object):
    """job subcommand entry
    """

    def __init__(self, parser=None, run=None):
        self.parser = parser or argparse.ArgumentParser(**kwargs)
        self.run = run
        self.help = None
        self.name = None

    def __call__(self, argv=None):
        namespace = self.parser.parse_args(argv)
        return self.run(namespace)

    def register(self, name, help=None):
        if name in jobs:
            warnings.warn("overwrite already registered job: "+name)
        self.name = name
        self.help = help
        jobs[name] = self

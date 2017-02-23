"""dummy model interface, as an example
"""
import os
import json
from runner.model import ModelInterface

class MyInterface(ModelInterface):
    """an example where the `json` file format is used for params and output I/O
    """

    def setup(self, rundir, params):
        json.dump(params, open(os.path.join(rundir, "params.json"),'w'), 
                  sort_keys=True, indent=2)


    def postprocess(self, rundir):
        return json.load(open(os.path.join(rundir, "params.json")))


    def command(self, rundir, params):
        if not self.args:
            return ['echo']  # just so that it does not fail
        else:
            return super(MyInterface, self).command(rundir, params)


# ModelInterface' first argument is a command, leave empty in this example
# since we overwrite "command"
copy = MyInterface('', work_dir="{}")

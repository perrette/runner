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


# ModelInterface' first argument is a command, leave empty in this example for 
# interactive use with `job run` (any new arguments will be appended)
mymodel = MyInterface('', work_dir="{}")

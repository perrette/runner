"""Job file example

---------------------------------------
"""

import os
from runner.core import Model, Job, Params

DEFAULT_PARAMS = 'params_example.json'

class MyModel(Model):
    """My model
    """
    def __init__(self, param_file):
        self.params = Params.read(param_file) 

    def setup_outdir(self, outdir):
        param_file = os.path.join(outdir, DEFAULT_PARAMS) # modified params
        self.params.write(param_file)
        # here you may copy/link necessary input file etc... or even call
        # an external script, or just do everything in the executable script.
        exe = os.path.join(outdir, "model.x")
        with open(exe, "w") as f:
            f.write('#!/bin/bash\n')
            f.write('echo "This is a dummy model"\n')
            f.write('echo params file: "$1"\n')
            f.write('echo outdir: "$2"\n')

        os.system("chmod +x {}".format(exe))

        cmd = param_file, outdir
        return exe, cmd

class MyJob(Job):

    def add_model_arguments(self, parser):
        " parser is a argparse.ArgumentParser instance, and more precisely, a group "
        parser.add_argument("--default-params", default=DEFAULT_PARAMS, help="default parameter file")

    def init_model(self, args):
        " args is the result of argparse.ArgumentParser.parse_args() "
        model = MyModel(args.default_params)  # this is the `__init__` function implemented above !
        return model

if __name__ == "__main__":
    job = MyJob(description=__doc__, outdir_default="out", epilog=
"""
Examples
--------

Check out the command line help:

    ./job --help 

Change one parameter (here let's assume `a` and `b` are acceptable parameters):

    ./job -p a=66

Now also test two different values for `b` (3 and 4):

    ./job -p a=66 b=3,4

Instead of having the program running in the terminal sequentially, make it happen
in the background, in a parallel way:

    ./job -p a=66 b=3,4 --background

Note: it is enougth to just write a bit of the parameter, such as `--back`
as long as it is unambiguous. `argparse` will deal with that.

Or just submit to the cluster:

    ./job -p a=66 b=3,4 --submit


You may run use `-i` for the interactive mode, write the parameter ensemble to
a file, or read a full parameter ensemble from file (combined from some outside script).
Just check out the the `--help`.

--------
""")
    job.parse_args_and_run()

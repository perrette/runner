runner
======
A package to run model simulations and explore parameter space.

Design philosophy
-----------------

`runner` basically basically helps you to build a light-weight wrapper around 
your model, as a python script, in order to easily run an ensemble of simulations. 
With time, it should grow in functionality with respect to parameter optimization 
and sensitivity analysis, while staying simple to use.

It provides:

- `Params` class(es), base class for parameters I/O (default formats: `json`, `nml`)
- `Model` class, as a wrapper around your executable (by default: initialized with one parameter file)
- `Job` class, which contains basically the command-line interface to initialize and run the model

Your model should consist in one executable file (compiled binary or bash script), 
and in the simplest case it may depend on one parameter file to read-in, 
and may require command-line parameters to run. More complex cases are also possible.

Take a look at the [example file](job_example.py), detailed below.


Wrap up your model: a simple case
---------------------------------

The first task, to do once and for all, consists in writing a wrapper for your model.
Let's consider the simple case described above, where your model only rely on
parameters file as argument. Let's say it uses the json format, such as below:

    [ 
    {
        "name": "a",
        "value": 4
    },
    {
        "name": "b",
        "value": 3.14
    }
    ]

This is the basic I/O format of the `Params` class in runner (see below for other formats).

Let's also assume that the model can be run via the executable `model.x` 
(this could be anything from a compiled C++ or Fortran program, or a bash or 
python script), and that both the path of the input parameter file and the 
output directory can be provided as input arguments:

    ./model.x params.json output

Just import the base `Model` class and overwrite the relevant methods:

```python
import os
from runner.core import Model

class MyModel(Model):
    """My model
    """
    def __init__(self, param_file):
        self.params = Params.read(param_file) 

    def setup_outdir(self, outdir):
        param_file = os.path.join(outdir, 'params.json') # modified params
        self.params.write(param_file)
        # here you may copy/link necessary input file etc... or even call
        # an external script, or just do everything in the executable script.
        exe = "model.x"  # executable
        cmd = param_file, outdir
        return exe, cmd
```

For now, that's it !

The `Model` class comes up with a few handy methods, check them out. Most useful are:

- `Model.update_params` : update model parameters (under the hood: calls `Params.update`)
- `Model.run` : run the model in the terminal or in the background
- `Model.submit` : submit the model to the cluster queue.

This may already be useful to play around with your model in python. 


Build the job script
--------------------
Here you will learn how to build a powerful command-line interface for parameter 
sensitivity studies. Let's assume your model is successfully wrapped 
(see simple case above, or more complex cases detailled in following sections).

You need to provide the default `Job` class with additional command-line arguments
required by your model, and to teach it how to initialize your model. You might 
need to take a look at `argparse` documentation...

```python
from runner.core import Job

class MyJob(Job):

    def add_model_arguments(self, parser):
        " parser is a argparse.ArgumentParser instance, and more precisely, a group "
        parser.add_argument("--default-params", default="params.json", help="default parameter file")

    def init_model(self, args):
        " args is the result of argparse.ArgumentParser.parse_args() "
        model = MyModel(args.default_params)  # this is the `__init__` function implemented above !
        return model
```

One caveat here: the arguments added in `add_model_arguments` must be distinct
from existing pre-defined arguments. So check out existing arguments by running
an empty Job, or play trial and error... Long names and usage will reduce the
chances of conflict.

The `Job.init_model` bit just initialize `Model` class from command-line arguments .

The last lines required to make the runner script are:
    
```python
if __name__ == "__main__":
    job = MyJob(description="some description", outdir_default="out")
    job.parse_args_and_run()
```

Note the description could also be passed `__doc__` 
if you bothered adding a python docstring to the file.

That is it for your runner script !


Try it out !
------------

Let's assume your runner script was written as an executable script as `job`
(in linux, you may simply add the header: `#!/usr/bin/env python2.7` 
and make the file executable `chmod +x job`). Otherwise, say if you instead
used wrote a normal python module such as `run_model.py`, just replace
`job` in the example below with `python run_model.py`:

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


Other parameter formats
-----------------------
A few alternative parameter formats are already defined, such as 
namelist (`runner.namelist.Namelist`). It is also possible to define 
new parameter formats, by simply subclassing the `Params` class
and subclassing the `Params.parse` and `Params.format` methods:

```python
from runner.params import Param, Params

class MyParams(Params):

    @classmethod
    def parse(cls, string):
        # convert the string (text file content) 
        # as a list of Param instances
        ...
        return cls(params)

    def format(self):
        # convert self (a list of Param instance) into 
        # a string which represents the content of a parameter file.
        ...
        return string
```

Check out the source code of the Namelist example for a practical implementation.

The two methods, parse and format, are called by `Params.read` and `Params.write`
so there is basically nothing more to do. Just initialize the Parameters
via `MyParams.read(file_name)` and do not forget to write `params.write()` in `Model.setup_outdir()`.


Several parameter files
-----------------------

The python trick consists in defining the combined parameters as just the 
concatenated list of the various subcomponents.

```python
class MyModel(Model):
    """My model
    """
    def __init__(self, param_file1, param_file2):
        self.params1 = Params.read(param_file1) 
        self.params2 = Params.read(param_file2) 
        
        # in case parameter names conflict, make sure the `module` attribute is set
        for p in self.params1:
            p.module = "mod1"
        for p in self.params2:
            p.module = "mod2"

        # modifying params will affect params1 and param2
        # since only the container change, but the Param instances are not copied
        self.params = self.params1 + self.params2

    def setup_outdir(self, outdir):
        self.params1.write(...)
        self.params2.write(...)
        exe = ...
        cmd = ...
        return exe, cmd
```

Custom parameter update
-----------------------
In some cases where the model parameters require some more elaborate 
cross-checking. This may be achieved ad-hoc in `Model.setup_outdir` by looking
at the internal of the `Params` instances, or, alternatively, by overloading
`Model.update_params`. 

TODO: update_params should take { name : value } pairs as arguments, for simplicity.
TODO: document `params_lookup` function.


Command-line parameters
-----------------------
In cases where your model takes command-line parameters as argument instead of 
a parameter file, you may still use `runner`. 
Currently you need to create a dummy parameter file with default parameter
values, so that `runner` can check that the proper parameter values are provided
to the script (see above example for the `json` format).

Then be careful to properly define command-line arguments as output of `Model.setup_outdir`.

TODO: Alternatively (but not yet implemented), the internal structure of `runner` could
be slighlty modified to remove the need for the `Params` class outside `Model`.
Currently, `Model.update_params` takes a `Params` instance as argument, but there is no
need for that. It could be just as fine to provide `Model.update_params` with a dictionary of
{ name : value }, and leave the model class deal with it freely (e.g. parameter
lookup and update in the standard case, or extend its `params` attribute in 
the command-line case without default parameters).

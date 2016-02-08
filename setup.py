# import os, sys, re
from distutils.core import setup
import versioneer


with open('README.md') as file:
    long_description = file.read()


# Actually important part
setup(name='runner',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author='Mahe Perrette',
      author_email='mahe.perrette@pik-potsdam.de',
      description='helper to run executables, submit to cluster, combine parameters etc',
      # keywords=('fortran','template','namelist'),
      # basic stuff here
      # py_modules = ['nml2f90'],
      packages = ['runner'],
      # package_data = {'nml2f90':['templates/*.f90', 'libraries/*f90']},
      # scripts = ['scripts/nml2f90', 'scripts/f2nml'],
      long_description=long_description,
      url='https://gitlab.com/greenrise/runner',
      # license = "MIT",
      )



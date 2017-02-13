from  distutils.core import setup
import versioneer

setup(name='runner',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author_email='mahe.perrette@pik-potsdam.de',
      packages = ['runner', 'runner.lib', 'runner.ext', 'runner.job'],
      depends = ['numpy', 'scipy', 'six', 'tox'],
      scripts = ['scripts/job'], 
      )

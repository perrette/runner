from  distutils.core import setup
import versioneer

setup(name='simtools',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author_email='mahe.perrette@pik-potsdam.de',
      packages = ['simtools', 'simtools.sampling', 'simtools.model'],
      scripts = ['job'], 
      )

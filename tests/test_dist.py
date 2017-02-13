from __future__ import absolute_import
import unittest
from scipy.stats import lognorm
from utils import runner

from runner.param import dist_todict, dist_fromkw


class TestDistScipy(unittest.TestCase):

    kw = {'loc': 10, 'name': 'lognorm', 'scale': 11, 'shapes': (2,)}

    def test_todict(self):
        self.assertEqual(dist_todict(lognorm(2, loc=10, scale=11)), self.kw)
        self.assertEqual(dist_todict(lognorm(2, 10, 11)), self.kw)
        self.assertEqual(dist_todict(lognorm(2, 10, scale=11)), self.kw)
        self.assertEqual(dist_todict(lognorm(2, loc=10, scale=11)), self.kw)

        default = {'loc': 0, 'name': 'lognorm', 'scale': 1, 'shapes': (2,)}
        self.assertEqual(dist_todict(lognorm(2, loc=0)), default)
        self.assertEqual(dist_todict(lognorm(2)), default)
        self.assertEqual(dist_todict(lognorm(2, 0)), default)

    def test_roundtrip(self):
        self.assertEqual(dist_todict(dist_fromkw(**self.kw)), self.kw)


class TestDistDiscrete(unittest.TestCase):

    kw = {'name':'discrete', 'values': [1,2,3]}

    def test_fromkw(self):
        d = dist_fromkw(**self.kw)
        self.assertIsInstance(d, runner.param.DiscreteDist)
        self.assertEqual(d.values.tolist(), self.kw['values'])

    def test_roundtrip(self):
        self.assertEqual(dist_todict(dist_fromkw(**self.kw)), self.kw)


class TestParam(unittest.TestCase):

    kw = {'name':'myparam', 
          'default': 4,
          'dist_name': 'lognorm', 
          'dist_loc': 10, 
          'dist_scale': 11, 
          'dist_shapes': (2,)}


if __name__ == '__main__':
    unittest.main()

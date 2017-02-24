from __future__ import absolute_import
import unittest
from scipy.stats import lognorm
from utils import runner

from runner.tools.dist import dist_todict, dist_fromkw
from runner.tools.dist import dist_todict2, dist_fromkw2, DiscreteDist
from runner.param import Param


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
        d = dist_fromkw2(**self.kw)
        self.assertIsInstance(d, DiscreteDist)
        self.assertEqual(d.values.tolist(), self.kw['values'])

    def test_roundtrip(self):
        self.assertEqual(dist_todict2(dist_fromkw2(**self.kw)), self.kw)


class TestParamIO(unittest.TestCase):
    def setUp(self):
        self.a = Param.parse('a=N?3,2')
        self.b = Param.parse('b=U?-1,1')

    def test_asdict(self):
        self.assertEqual(self.a.as_dict(), {
            'name':'a',
            'dist_name':'norm',
            'dist_loc':3,
            'dist_scale':2,
        })
    
        self.assertEqual(self.b.as_dict(), {
            'name':'b',
            'dist_name':'uniform',
            'dist_loc':-1,
            'dist_scale':2,
        })

    def test_roundtrip(self):
        self.assertEqual(Param.fromkw(**self.a.as_dict()), self.a)
        self.assertEqual(Param.fromkw(**self.b.as_dict()), self.b)


if __name__ == '__main__':
    unittest.main()

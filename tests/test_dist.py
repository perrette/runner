from __future__ import absolute_import
import unittest
from scipy.stats import lognorm
from utils import runner

from runner.tools import dist_todict, dist_fromkw

class TestDist(unittest.TestCase):

    default = {'loc': 0, 'name': 'lognorm', 'scale': 1, 'shapes': (2,)}
    variation = {'loc': 10, 'name': 'lognorm', 'scale': 11, 'shapes': (2,)}

    def test_todict(self):
        self.assertEqual(dist_todict(lognorm(2)), self.default)
        self.assertEqual(dist_todict(lognorm(2, 0)), self.default)
        self.assertEqual(dist_todict(lognorm(2, 0, 1)), self.default)
        self.assertEqual(dist_todict(lognorm(2, 0, scale=1)), self.default)
        self.assertEqual(dist_todict(lognorm(2, loc=0)), self.default)
        self.assertEqual(dist_todict(lognorm(2, loc=0, scale=1)), self.default)
        self.assertEqual(dist_todict(lognorm(2, loc=10, scale=11)), self.variation)
        self.assertEqual(dist_todict(lognorm(2, 10, 11)), self.variation)


if __name__ == '__main__':
    unittest.main()

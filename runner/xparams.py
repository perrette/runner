# XParams
from collections import OrderedDict as odict
from runner.tools import DataFrame
from runner.resample import Resampler, RESAMPLING_METHOD, NEFF_BOUNDS


# Ensemble parameters
# NOTE: this XParams class could well be in another module and be simply imported,
# but for now it makes one module less...
class XParams(DataFrame):
    """Experiment params
    """
    def __init__(self, values, names, default=None):
        self.values = values 
        self.names = names
        self.default = default

    def pset_as_array(self, i=None):
        if i is None:
            pvalues = self.default
        else:
            pvalues = self.values[i]
        return pvalues

    def pset_as_dict(self, i=None):
        """return parameter set as a dictionary
        """
        pvalues = self.pset_as_array(i)

        if pvalues is None:
            return odict()  # case were default parameters are not provided

        params = odict()
        for k, v in zip(self.names, pvalues):
            params[k] = v
        return params

    def resample(self, weights, size=None, seed=None, method=RESAMPLING_METHOD, 
                 iis=False, epsilon=None, neff_bounds=NEFF_BOUNDS, bounds=None):
        """
        Parameters
        ----------
        weights : array of weights (must match params' size)
        size : new ensemble size, by default same as current
        seed : random state seed (None)
        method : method for weighted resampling (see runner.resample.Resampler)
        iis : step of the Iterative Importance Sampling strategy (Hannan and Hargreave)
            where weights are flattened (epsilon exponent) and jitter (noise) is added
            to the resampled ensemble, as a fraction epsilon of its (weighted) 
            covariance. In the linear case, the combination of flattened resampling
            and jitter addition is equivalent to one time resampling with full weights.
        epsilon : scaling exponent for the weights, ie `weights**epsilon` [iis method only] 
            If not provided, epsilon is automatically generated to yield an effective
            ensemble size comprised in the neff_bounds range. Starting value: epsilon.
        neff_bounds : target effective ensemble size to determine epsilon automatically
        bounds : authorized parameter range (experimental). If jitter addition yields parameters
            outside the specified range, try again a number of times. [iis method only]


        Returns
        -------
        XParams instance
        """
        if weights.size != self.size:
            raise ValueError("params and weights size do not match")

        resampler = Resampler(weights) # default size implied by weights
        if iis:
            vals = resampler.iis(self.values, 
                           size=size, seed=seed, method=method, 
                           bounds=bounds, neff_bounds=neff_bounds, 
                           epsilon=epsilon)

        else:
            idx = resampler.sample(size=size, seed=seed, method=method)
            vals = self.values[idx]
        return XParams(vals, self.names)

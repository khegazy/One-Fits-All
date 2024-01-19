from typing import Any
import numpy as np
import numpy.random as rnd
from functools import partial

class NoiseTransform:
    def __init__(self, drift, var, var_scale) -> None:
        fxn_dict = {
            'sqrt_data' : self.sqrt_data
        }
        
        if drift is None or drift.lower() == 'identity':
            self.drift = self.identity
        else:
            raise NotImplemented
        
        if var is None:
            self.var = lambda x : 0
        else:
            self.var = partial(fxn_dict[var], var_scale)

    def identity(self, data):
        return data
    
    def sqrt_data(self, scale, data):
        return scale*np.sqrt(data)


class Poisson(NoiseTransform):
    def __init__(self, drift, var, var_scale) -> None:
        super().__init__(drift, var, var_scale)
    
    def __call__(self, data):
        return rnd.poisson(self.drift(data), data.shape)


class Gaussian(NoiseTransform):
    def __init__(self, drift, var, var_scale) -> None:
        super().__init__(drift, var, var_scale)
    
    def __call__(self, data) -> Any:
        return self.drift(data) + rnd.normal(0, np.sqrt(self.var(data)), data.shape)
    
noise_dict = {
    'poisson' : Poisson,
    'gaussian' : Gaussian,
    'normal' : Gaussian
}


def get_transform(transform_name, drift, var, var_scale):
    return noise_dict[transform_name.lower()](drift, var, var_scale)


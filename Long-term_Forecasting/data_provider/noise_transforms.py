from typing import Any
import numpy as np
import numpy.random as rnd
from functools import partial

class NoiseTransform:
    def __init__(self, drift, var, std_scale) -> None:
        fxn_dict = {
            'identity' : self.identity,
            'constant' : self.constant,
            'sqrt_data' : self.sqrt_data
        }
        self.std_scale = std_scale

        if drift is None or drift.lower() == 'identity':
            self.drift = self.identity
        else:
            raise NotImplemented
        
        if var is None:
            self.var = lambda x : 0
        else:
            self.var = fxn_dict[var]

    def identity(self, data):
        return data
    
    def constant(self, data):
        return np.ones_like(data)
    
    def sqrt_data(self, data):
        return np.sqrt(data)


class Uniform(NoiseTransform):
    def __init__(self, drift, var, std_scale) -> None:
        super().__init__(drift, var, std_scale)
    
    def __call__(self, data):
        return self.drift(data) + self.std_scale*rnd.uniform(0, 1, data.shape)


class Poisson(NoiseTransform):
    def __init__(self, drift, var, std_scale) -> None:
        super().__init__(drift, var, std_scale)
    
    def __call__(self, data):
        return rnd.poisson(self.drift(data), data.shape)


class Gaussian(NoiseTransform):
    def __init__(self, drift, var, std_scale) -> None:
        super().__init__(drift, var, std_scale)
    
    def __call__(self, data) -> Any:
        #print("NOISE INP DATA", data[1:10])
        scale = self.std_scale*np.sqrt(self.var(np.abs(data - np.amin(data, axis=-2, keepdims=True))))
        out = self.drift(data) + scale*rnd.normal(0, 1., data.shape)
        #print("NOISE OUT DATA", out[1:10])
        #print("NOISE SCALE", self.std_scale, scale[1:10])
        #print("DATA OUT NAN", out.shape, np.sum(np.isnan(out)))
        return out
    
noise_dict = {
    'uniform' : Uniform,
    'poisson' : Poisson,
    'gaussian' : Gaussian,
    'normal' : Gaussian
}


def get_transform(transform_name, drift, var, var_scale):
    return noise_dict[transform_name.lower()](drift, var, var_scale)


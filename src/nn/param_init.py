import numpy as np

def normal(arr: np.ndarray, mean=0, std=0.01):
    """ Apply normal distribution values to weights """

    rng = np.random.default_rng()
    arr[:] = rng.normal(loc=mean, scale=std, size=arr.shape)
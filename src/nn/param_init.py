import numpy as np

def normal(arr: np.ndarray, mean=0, std=0.01):
    """ Apply normal distribution values to weights """

    rng = np.random.default_rng()
    arr[:] = rng.normal(loc=mean, scale=std, size=arr.shape)

def kaiming(arr: np.ndarray):
    if arr.ndim == 2:
        fan_in = arr.shape[0]
    elif arr.ndim == 4:
        fan_in = arr.shape[1] * arr.shape[2] * arr.shape[3]
    else:
        raise ValueError("Kaiming initialization requires 2D or 4D weights")

    normal(arr, mean=0, std=np.sqrt(2.0 / fan_in))


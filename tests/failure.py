
import numpy as np
import xarray as xr
from itertools import repeat, product
from operator import mul
from time import clock
from functools import reduce
import pytest


from lerp import Mesh


def random_sort(arr):
    for i in range(arr.ndim):
        arr = np.sort(arr, axis=i)
    return arr


np.random.seed(123)

nz = 100


m4d = xr.DataArray(data=random_sort(
    np.random.exponential(size=(4, 5, nz)) * 1000),
    coords=[('x', [1, 2, 3, 6]),
            ('y', [13, 454, 645, 1233, 1535]),
            ('z', np.sort(np.random.uniform(0, 200, size=(nz,))))])

mesh4d = Mesh(m4d)


mymesh = mesh4d[2:, :, :]

#with pytest.raises(ValueError):
mymesh(mymesh.x.values, mymesh.y.values)

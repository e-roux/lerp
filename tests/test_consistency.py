
import numpy as np
import xarray as xr
from lerp import Mesh
import sys

def random_sort(arr):
    for i in range(arr.ndim):
        arr = np.sort(arr, axis=i)
    return arr


nz = 100

m4d = xr.DataArray(data=random_sort(
    np.random.exponential(size=(4, 5, nz)) * 1000),
    coords=[('x', [1, 2, 3, 6]),
            ('y', [13, 454, 645, 1233, 1535]),
            ('z', np.sort(np.random.uniform(0, 200, size=(nz,))))])

mesh4d = Mesh(m4d)


# def test_consistency_at_breakpoints():
mymesh = mesh4d  # [2:, :, :]
for _ix, _x in enumerate(mymesh.x.values):
    for _iy, _y in enumerate(mymesh.y.values):
        for _iz, _z in enumerate(mymesh.z.values):
            # print(f"{_ix+1}/{_iy+1}/{_iz+1}",
            #       1 + _iz + (_iy + _ix * len(mymesh.y)) * len(mymesh.z), "/", mymesh.data.size,
            #       "-", sys.getrefcount(mymesh), mymesh(_x, _y, _z),
            #       mymesh[_ix, _iy, _iz].values)
            mymesh(_x, _y, _z)


def test_consistency_at_breakpoints():
    mymesh = mesh4d  # [2:, :, :]
    for _ix, _x in enumerate(mymesh.x.values):
        for _iy, _y in enumerate(mymesh.y.values):
            for _iz, _z in enumerate(mymesh.z.values):
                assert mymesh[_ix, _iy, _iz].values == mymesh(_x, _y, _z)[0]


def test_simple():
    np.random.seed(123)
    m3d = Mesh(coords=[('x', [1, 2, 3, 6]),
                       ('y', [13, 454, 645, 1233, 1535])],
               data=np.random.randn(4, 5))

    res = m3d.interpolation([1.2, 5.6, 6], [645] * 3)

    assert res[0] == 0.1406002726703582

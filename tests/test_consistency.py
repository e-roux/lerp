
import numpy as np
import xarray as xr
from lerp import Mesh


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


def test_consistency_at_breakpoints():
    mymesh = mesh4d  # [2:, :, :]
    for _ix, _x in enumerate(mymesh.x.values):
        for _iy, _y in enumerate(mymesh.y.values):
            for _iz, _z in enumerate(mymesh.z.values):
                assert mymesh[_ix, _iy, _iz].values == mymesh(_x, _y, _z)


def test_simple():
    np.random.seed(123)
    m3d = Mesh(coords=[('x', [1, 2, 3, 6]),
                       ('y', [13, 454, 645, 1233, 1535])],
               data=np.random.randn(4, 5))

    res = m3d.interpolation([1.2, 5.6, 6], [645] * 3)

    assert res[0] == 0.1406002726703582


# pytest.fail("not configu")

# with pytest.raises(AssertionError, message="Expecting ZeroDivisionError"):
#    True
# simple_check(mymesh)
# t1 = clock()
# t2 = clock()
# print(f"Test ok, performed in {t2 - t1}s")
# t3 = clock()
# mymesh(*[np.ravel(list(repeat(mymesh.coords[elt].values,
#                              reduce(mul, [len(mymesh[_s])
#        for _s in set(mymesh.dims) - set(elt)])))) for elt in mymesh.dims])
# t4 = clock()
# print(f"Test ok, performed in {t4 - t3}s")

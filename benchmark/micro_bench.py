# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

from numpy import (random, array, arange, linspace, interp)
from lerp import Mesh
from time import time
import numpy as np
import pandas as pd
import sys

from lerp.core.interpolation import my_interp


def tiny_bench():

    x = np.linspace(0, 20_000 * np.pi, 100_000)
    y = np.sin(x)

    m2d = Mesh(coords=[('x', x)], data=y)
    x = m2d.x.data
    y = m2d.data
    # print(y[1])

    results = {}
    _range = np.arange(0, 1_050_000, 50_000)

    for N in _range:
        _xi = np.random.randint(1, 10000, N).astype(
            np.float64) + np.random.random(N)
        _xi.sort()
        t1 = time()
        # print(f"Références à y: {sys.getrefcount(y)}")
        # print(f"Références à _xi: {sys.getrefcount(_xi)}")
        # print(f"Références à x: {sys.getrefcount(x)}")
        res1 = m2d.interpolation(_xi, interp='linear', extrap='hold')
        # print(f"Références à y: {sys.getrefcount(y)}")
        t2 = time()

        # print(f"Références à _xi: {sys.getrefcount(_xi)}")
        # print(f"Références à x: {sys.getrefcount(x)}")
        res2 = my_interp(_xi, x, y)
        t3 = time()
        results[N] = [t1, t2, t3]

    all_runs = pd.DataFrame(results) * 1000
    all_runs = all_runs.T.diff(axis=1).loc[:, 1:]
    all_runs.columns = ["Mesh", "Numpy"]
    all_runs.index.name = "Interpolated array size"

    return all_runs


print("*" * 60)
print("Tiny bench")
print("*" * 60)
print(tiny_bench())


def random_sort(arr):
    for i in range(arr.ndim):
        arr = np.sort(arr, axis=i)
    return arr


np.random.seed(123)
m4d = Mesh(coords=[('x', [1, 2, 3, 6]),
                   ('y', [13, 454, 645, 1233, 1535]),
                   ('z', np.sort(np.random.uniform(0, 200, size=(20,))))],
           data=random_sort(np.random.exponential(size=(4, 5, 20)) * 1000))
# print(m4d.x.dtype)
# print(m4d([6, 3, 3], [2, 2, 4], [3, 9, 3]))
# print(m4d(1,  13, 3))

# print(m4d[1])

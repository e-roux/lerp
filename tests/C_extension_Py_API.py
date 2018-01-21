# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

#from numpy import (random, array, arange, linspace, interp)
import numpy as np
from lerp.core import interp
from time import time
import pandas as pd

from lerp import Mesh

#A = linspace(0.1, 291381723, 10_000_001)

#res = interp.interpolation(A, A, "aze")

#print(res)


def tiny_bench():

    x = np.linspace(0, 200 * np.pi, 1000) - np.random.random(1000)
    x.sort()
    y = np.sin(x) + np.random.random(1000)

    # print(f"là {y[3]}")
    #    m2d = Mesh(x,y)
 #   x = m2d.x.data
  #  y = m2d.data

    results = {}
    _range = np.arange(0,1_050_000, 50_000)
    _range = [100_000]

    for N in _range:
        _xi = np.random.randint(1, 10000, N).astype(np.float64) + np.random.random(N)

        myMesh = Mesh(x, y)
        res = np.empty_like(_xi)
        _xi.sort()
        t1 = time()
        # print(_xi[:3])
        #res1 = interp.interpolation(y, [x], [_xi],  "linear", extrap='linear')
        res1 = myMesh.interpolation(_xi, interp='linear', extrap='hold')
#        m2d.interpolation(_xi, interp='linear', extrap='hold')
        t2 = time()
        res2 = np.interp(_xi, x, y)
        #x.searchsorted(_xi)
        t3 = time()
        results[N] = [t1, t2, t3]
        assert np.all(np.isclose(res1,res2)), "Interpolation do not give the same results."


    # print(f"{x[2:4]} - {y[2:4]} - {_xi[2:4]}")
    # print(f"{res1[2:4]} - {res2[2:4]}")
 
    all_runs = pd.DataFrame(results) * 1000
    all_runs = all_runs.T.diff(axis=1).loc[:,1:]
    all_runs.columns = ["Mesh", "Numpy"]
    all_runs.index.name = "Interpolated array size"

    return all_runs

print("*"*80)
print("Tiny bench")
print("*"*80)
print(tiny_bench())

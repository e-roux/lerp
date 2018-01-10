# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

from numpy import (random, array, arange, linspace, interp)
from lerp.mesh import Mesh
from time import time
import numpy as np
import pandas as pd


def test_success():
    assert True


def test():
    np.random.seed(123)
    m3d = Mesh(x=[1, 2, 3, 6], y=[13, 454, 645, 1233, 1535],
               data=np.random.randn(4, 5),
               label="le label")

    print(f"m3d : {m3d}")

    res = m3d.interpolation([1.2, 5.6, 6], 645)

    assert res[0] == 0.1406002726703582, f"res[0] = {res[0]} instead of 0.14060027"  #  2.09406656  2.18678609
    print(f"{res} for x=[1.2, 5.6, 6] and y=645")
    return res

# @profile
def tiny_bench():

    x = np.linspace(0, 2 * np.pi, 10)
    y = np.sin(x)

    m2d = Mesh(x,y)
    x = m2d.x.data
    y = m2d.data

    results = {}
    _range = np.arange(0,1_050_000, 50_000)

    for N in _range:
        _xi = np.random.randint(1, 10000, N).astype(np.float64) + np.random.random(N)
        _xi.sort()
        t1 = time()
        m2d.interpolation(_xi)
        t2 = time()
        interp(_xi, x, y)
        t3 = time()
        results[N] = [t1, t2, t3]

    all_runs = pd.DataFrame(results) * 1000
    all_runs = all_runs.T.diff(axis=1).loc[:,1:]
    all_runs.columns = ["Mesh", "Numpy"]

#    all_runs = pd.DataFrame(data={c: [_r / r.loops for _r in r.all_runs] for c, r in zip(_range, results)})
    return all_runs



x = np.linspace(0, 2 * np.pi, 10)
y = np.sin(x)
xi = np.linspace(-1.5,  2 * np.pi + 1.5, 1000)
m2d = Mesh(x,y)

print(m2d.derivate(xi))

print("*"*80)
print("test")
print("*"*80)
test()
print("*"*80)
print("Tiny bench")
print("*"*80)
print(tiny_bench())

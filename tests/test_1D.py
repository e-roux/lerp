# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

from numpy import (random, array, arange, linspace, interp)
from lerp import Mesh
from time import time
import numpy as np
import pandas as pd


x = np.linspace(0, 2 * np.pi, 10)
y = np.sin(x)
xi = np.linspace(-1.5,  2 * np.pi + 1.5, 10)
m2d = Mesh(coords=[('x', x)],data=y)

print("*"*30)
for _x in m2d.rolling(x=1):
    print(f"{_x[0].x.data:.2f}, {_x[1].data[0]:.2f}")
print("*"*30)

for _x in xi:
    print(_x, m2d([_x]))

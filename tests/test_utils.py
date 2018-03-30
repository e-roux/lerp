# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import sys
from numpy import (random, array, arange, linspace, interp)
from lerp import Mesh
from lerp.core import utils
from time import time
import numpy as np
import pandas as pd


x = np.linspace(0, 2 * np.pi, 10)
y = np.sin(x)
xi = np.linspace(-1.5, 2 * np.pi + 1.5, 1000)
m2d = Mesh(coords=[('x', x)], data=y)

print(y.shape)

print(f"Références à y: {sys.getrefcount(y)}")

a = utils.testmethod(y)

print(f"Références à y: {sys.getrefcount(y)}")

print(a.shape)

del a

print(f"Références à y: {sys.getrefcount(y)}")

print("Terminé")
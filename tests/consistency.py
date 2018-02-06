
import numpy as np
import xarray as xr
from itertools import repeat, product
from operator import mul
from time import clock
from functools import reduce


from lerp import Mesh

def random_sort(arr):
    for i in range(arr.ndim):
        arr = np.sort(arr, axis=i)
    return arr

np.random.seed(123)

nz = 100
m4d = xr.DataArray(data=random_sort(np.random.exponential(size=(4, 5, nz))*1000),
                   coords=[('x', [1, 2, 3, 6]),
                           ('y', [13, 454, 645, 1233, 1535]),
                           ('z', np.sort(np.random.uniform(0, 200, size=(nz,))))])
mesh4d = Mesh(m4d)



#print(mesh4d(1, 13, 33.013288565607482))

#print("1er: ", mesh4d(2, 1233, 73.7749683362))

mymesh = mesh4d[2:,:,:]

t1 = clock()

for _ix, _x in enumerate(mymesh.x.values):
    for _iy, _y in enumerate(mymesh.y.values):
        for _iz, _z in enumerate(mymesh.z.values):
#            print(_ix, _iy, _iz)
#           print("Sl", mymesh[_ix, _iy, _iz].values)
#           print("Int", mymesh(_x, _y, _z)) 
            #pass
            assert mymesh[_ix, _iy, _iz].values == mymesh(_x, _y, _z), f"Problème à {_x}, {_y}, {_z}"
 
#for i, ind in enumerate(product(*[mesh4d.coords[k].values for k in mesh4d.coords])):
#   print(i, *ind)
#    print(mesh4d(*ind))
t2 = clock()
print(f"Test ok, performed in {t2 - t1}s")
t3 = clock()
mymesh(*[np.ravel(list(repeat(mymesh.coords[elt].values,
                             reduce(mul, [len(mymesh[_s])
             for _s in set(mymesh.dims) - set(elt)])))) for elt in mymesh.dims])
t4 = clock()
print(f"Test ok, performed in {t4 - t3}s")
#print(mymesh[-1, -1, -1].values)
#print(repeat(mymesh.x.values), 

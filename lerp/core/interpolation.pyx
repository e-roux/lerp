# -*- coding: utf-8 -*-
"""This module delivers utilities to interpolate dataArray
calling C-code with ctypes.


http://calcul.math.cnrs.fr/Documents/Ecoles/2013/python/CythonInterface.pdf
"""

import numpy as np
cimport numpy as np
cimport cython

ctypedef struct ndtable:
    int shape[32]
    int strides[32]
    int ndim
    double *data
    int resultSize
    int itemsize
    int *breakpoints[32]

cdef enum NDTable_InterpMethod_t:
    NDTABLE_INTERP_HOLD = 1
    NDTABLE_INTERP_NEAREST
    NDTABLE_INTERP_LINEAR
    NDTABLE_INTERP_AKIMA
    NDTABLE_INTERP_FRITSCH_BUTLAND
    NDTABLE_INTERP_STEFFEN

cdef enum NDTable_ExtrapMethod_t:
    NDTABLE_EXTRAP_HOLD = 1
    NDTABLE_EXTRAP_LINEAR
    NDTABLE_EXTRAP_NONE

cdef extern from "NDTable.h":
    void evaluate_interpolation(ndtable table,
                                const double **params,
                                int ndims,
                                NDTable_InterpMethod_t interp_method,
                             	NDTable_ExtrapMethod_t extrap_method,
                             	int nvalues,
                             	double *values)

def interpolation(double[:, ::1] self not None, *points, interp='linear', extrap='hold', **kwargs):
    """Interpolation
    """

    assert len(set(self.dims) & set(kwargs)) + len(points) == self.ndim, \
        "Not enough dimensions for interpolation"

    # First:
    #   - convert points (tuple) to list,
    #   - clean-up arguments in case: mix usage points/kwargs
    #   - create a clean argument dict
    points = list(points)

    args = {_x : kwargs[_x] if _x in kwargs else points.pop(0)
            for _x in self.dims}

    # Compute args dimensions and check compatibility without
    # broadcasting rules.
    dims = np.array([len(args[_k]) if "__len__" in dir(args[_k])
                     else 1 for _k in args])
    assert all((dims == max(dims)) + (dims == 1)), "problème"

    args = [np.asarray(args[_x], 'f8')
            if "__len__" in dir(args[_x])
            else np.ones((max(dims),), 'f8') * args[_x]
            for _x in self.dims]

    values = np.empty(args[0].shape)

    # Memoryviews
    # http://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html

    shape = np.array(self.data.shape, dtype=int)
    shape.resize((32, 1), refcheck=False)

    strides = np.array(self.data.strides, dtype=int)
    strides.resize((32, 1), refcheck=False)

    data = seld.data.__array_interface__['data']

    #cdef double narr_view = self.data
    #cdef np.ndarray[double, mode="c"] data
    # cdef double[:] * data = np.ascontiguousarray(self.data, dtype=np.double)
    #data = np.ascontiguousarray(self.data, dtype=ctypes.c_double)
    #cdef double* _ptr
    #_ptr = &data[0]
    # cdef double* dptr = <double *>np.PyArray_DATA(self.data)

    a = ndtable(shape,
                strides,
                self.data.ndim,
                data,  # np.PyArray_DATA(self.data),
                self.data.size,
                self.data.itemsize
                )

    # res = ctypes_interpolation(NDTable_t(data=self),
    #                             c_params_p(*[_a.ctypes.get_as_parameter()
    #                                           for _a in args]),
    #                              ctypes.c_int(self.ndim),
    #                              INTERP_METH[interp],
    #                              EXTRAP_METH[extrap],
    #                              ctypes.c_int(values.size),
    #                              values.ctypes.get_as_parameter())

    res = 0  # evaluate_interpolation(a)
    assert res == 0, 'An error occurred during interpolation'

    return values[0] if len(values) == 1 else values


def derivate(self, *points, interp='linear', extrap='hold', **kwargs):
    """derivate
    """
    pass

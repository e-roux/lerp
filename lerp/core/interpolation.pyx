# -*- coding: utf-8 -*-
"""This module delivers utilities to interpolate dataArray
calling C-code with ctypes.
"""

import ctypes
from enum import IntEnum
from numpy.ctypeslib import (ndpointer, load_library)
import numpy as np
from lerp.core import path2so

class LookUpEnum(IntEnum):
    """LookUpEnum provide from_param method for ctypes.
    """
    @classmethod
    def from_param(cls, obj):
        return int(obj)

INTERP_METH = LookUpEnum('INTERP_METH',
                         'hold nearest linear akima fritsch_butland steffen')
EXTRAP_METH = LookUpEnum('EXTRAP_METH',
                         'hold linear')
MAX_NDIMS = 32
ARRAY_MAX_NDIMS = ctypes.c_int * MAX_NDIMS
POINTER_TO_DOUBLE = ndpointer(dtype='f8', flags='C_CONTIGUOUS')
POINTER_TO_BP = POINTER_TO_DOUBLE * MAX_NDIMS

class NDTable_t(ctypes.Structure):
    """
    Parameter : Mesh object
    """
    _fields_ = [("shape", ctypes.c_int * MAX_NDIMS),
                ("strides", ctypes.c_int * MAX_NDIMS),
                ("ndim", ctypes.c_int),
                ("data", POINTER_TO_DOUBLE),
                ("size", ctypes.c_int),
                ("itemsize", ctypes.c_int),
                ("breakpoints", POINTER_TO_BP)]

    def __init__(self, *args, **kwargs):
        if 'data' in kwargs:
            _mesh = kwargs['data']
            data = _mesh.data.astype('f8')
            kwargs['data'] = data.ctypes.data_as(POINTER_TO_DOUBLE)
            kwargs['shape'] = ARRAY_MAX_NDIMS(*data.shape)
            kwargs['strides'] = ARRAY_MAX_NDIMS(*data.strides)
            kwargs['itemsize'] = data.itemsize
            kwargs['ndim'] = data.ndim
            kwargs['size'] = data.size
            kwargs['breakpoints'] = POINTER_TO_BP(*[np.asanyarray(_mesh.coords[elt], \
                          dtype='f8', order='C').ctypes.data \
                          for elt in _mesh.dims])

        super(NDTable_t, self).__init__(*args, **kwargs)

    @classmethod
    def from_param(cls, obj):
        return ctypes.byref(obj)

################################################################################
# Import evaluate_interpolation
################################################################################
libNDTable = load_library('libNDTable', path2so)

evaluate_interpolation = libNDTable.evaluate_interpolation
evaluate_interpolation.argtypes = [NDTable_t, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_void_p]
evaluate_interpolation.restype = ctypes.c_int

evaluate_derivative = libNDTable.evaluate_derivative
evaluate_derivative.argtypes = [NDTable_t, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p]
evaluate_derivative.restype = ctypes.c_int


def interpolation(self, *points, interp='linear', extrap='hold', **kwargs):
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

    c_params_p = ctypes.c_void_p * len(self.dims)

    res = evaluate_interpolation(NDTable_t(data=self),
                                 c_params_p(*[_a.ctypes.get_as_parameter()
                                              for _a in args]),
                                 ctypes.c_int(self.ndim),
                                 INTERP_METH[interp],
                                 EXTRAP_METH[extrap],
                                 ctypes.c_int(values.size),
                                 values.ctypes.get_as_parameter())

    assert res == 0, 'An error occurred during interpolation'

    return values[0] if len(values) == 1 else values


def derivate(self, *points, interp='linear', extrap='hold', **kwargs):
    """derivate
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

    dxi = [np.ones_like(_x) for _x in args]

    values = np.empty(args[0].shape)

    c_params_p = ctypes.c_void_p * len(self.dims)

    res = evaluate_derivative(NDTable_t(data=self),
                              c_params_p(*[_a.ctypes.get_as_parameter()
                                           for _a in args]),
                              c_params_p(*[_a.ctypes.get_as_parameter()
                                           for _a in dxi]),
                              ctypes.c_int(self.ndim),
                              INTERP_METH[interp],
                              EXTRAP_METH[extrap],
                              ctypes.c_int(values.size),
                              values.ctypes.get_as_parameter())

    assert res == 0, 'An error occurred during interpolation'

    return values[0] if len(values) == 1 else values

#cdef extern from "NDTable.h":
#    void evaluate_interpoalation(int tons)

# cdef struct ndtable:
#     int shape[MAX_NDIMS]
#     int ndim


#	int 	shape[MAX_NDIMS]    # Array of data array dimensions.
#	int 	strides[MAX_NDIMS]  # bytes to step in each dimension when
								# traversing an array.
#	int		ndim			    # Number of array dimensions.
#	double *data			    # Buffer object pointing to the start
								# of the array’s data.
#	int		size			    # Number of elements in the array.
#	int     itemsize		    # Length of one array element in bytes.
#	double *breakpoints[MAX_NDIMS]  # array of pointers to the scale values

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


# Note: recipe #15.1
# Python Cookbook, D. Beazley
# O'Reilly
# Define a special type for the 'double *' argument
# The important element is from_param
class DoubleArrayType:
    def from_param(self, param):
        typename = type(param).__name__
        if hasattr(self, 'from_' + typename):
            return getattr(self, 'from_' + typename)(param)
        elif isinstance(param, ctypes.Array):
            return param
        else:
            raise TypeError("Can't convert %s" % typename)

    # Cast from array.array objects
    def from_array(self, param):
        if param.typecode != 'd':
            raise TypeError('must be an array of doubles')
        ptr, _ = param.buffer_info()
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))

    # Cast from lists/tuples
    def from_list(self, param):
        val = ((ctypes.c_double)*len(param))(*param)
        return val

    from_tuple = from_list

    # Cast from a numpy array
    def from_ndarray(self, param):
        return param.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

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

# -*- coding: utf-8 -*-
"""
This module delivers utilities to manipulate data meshes

"""
import sys

from ctypes import *
from os.path import (dirname, join as pjoin)
from numpy.ctypeslib import (ndpointer, load_library)
from enum import IntEnum
import numpy as np

__all__ = ["INTERP_METH", "EXTRAP_METH", "libNDTable",
           "_interpol", "_derivate"]

class LookUpEnum(IntEnum):
    @property
    def ctype(self):
        return c_int(self.value)

INTERP_METH = LookUpEnum('INTERP_METH',
                         'hold nearest linear akima fritsch_butland steffen')
EXTRAP_METH = LookUpEnum('EXTRAP_METH',
                         'hold linear')

path2so = dirname(__file__)
#if path2so not in sys.path:
#    sys.path.append(path2so)
libNDTable = load_library('libNDTable', path2so)

MAX_NDIMS = 32
ARRAY_MAX_NDIMS = c_int * MAX_NDIMS
POINTER_TO_DOUBLE = ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')

class NDTable_t(Structure):
    """
    Parameter : Mesh object
    """
    _fields_ = [("shape", c_int * MAX_NDIMS),
                ("strides", c_int * MAX_NDIMS),
                ("ndim", c_int),
                ("data", POINTER_TO_DOUBLE),
                ("size", c_int),
                ("itemsize", c_int),
                ("breakpoints", POINTER_TO_DOUBLE * MAX_NDIMS)]

    def __init__(self, *args, **kwargs):
        if 'data' in kwargs:
            _mesh = kwargs['data']
            try:
                data = _mesh.d.astype(np.float64)
            except AttributeError:
                data = _mesh.data.astype(np.float64)

            kwargs['data'] = data.ctypes.data_as(POINTER_TO_DOUBLE)
            kwargs['shape'] = ARRAY_MAX_NDIMS(*data.shape)
            kwargs['strides'] = ARRAY_MAX_NDIMS(*data.strides)
            kwargs['itemsize'] = data.itemsize
            kwargs['ndim'] = data.ndim
            kwargs['size'] = data.size
            breakpoints_ = (POINTER_TO_DOUBLE * MAX_NDIMS)()

            try:
                breakpoints = (np.asanyarray(getattr(_mesh, elt),
                                             dtype=np.float64, order='C')
                               for elt in _mesh.axes)
            except AttributeError:
                breakpoints = (np.asanyarray(getattr(_mesh, elt),
                                             dtype=np.float64, order='C')
                               for elt in _mesh.coords)

            for i, scale in enumerate(breakpoints):
                breakpoints_[i] = scale.ctypes.data
            kwargs['breakpoints'] = breakpoints_


        super(NDTable_t, self).__init__(*args, **kwargs)

################################################################################
# Import evaluate_struct
################################################################################
evaluate_struct = libNDTable.evaluate_struct
evaluate_struct.argtypes = [POINTER(NDTable_t), c_void_p, c_int,
                            c_int, c_int, c_int, c_void_p]
evaluate_struct.restype = c_int

def _interpol(data, coordinates, interp='linear', extrap='hold'):
    # Values to interpolate
    # if all([p[i] for p in points]):

    #---------------------------------------------------------

#     points = []
#     if data.ndim == 1:
#         points = [np.asarray([*coordinates]).ravel()]
# #                  for  i, _ in enumerate(coordinates)]
#     elif data.ndim > 1:
#         for i, _ in enumerate(coordinates):
#             points.append(np.asarray(coordinates[i], np.float64))
#
#         #points = [np.asarray([*i]).ravel()
#         #          for  i in zip(*coordinates)]
#     else:
#         print("Wrong!")

    points = coordinates


    values = np.empty(points[0].shape)
    # params
    params = (c_void_p * len(points))()
    for i, param in enumerate(points):
        params[i] = param.ctypes.get_as_parameter()
    res = evaluate_struct(byref(NDTable_t(data=data)),
                          params,
                          c_int(len(params)),
                          c_int(INTERP_METH[interp]),
                          c_int(EXTRAP_METH[extrap]),
                          c_int(values.size),
                          values.ctypes.get_as_parameter()
                          )
    assert res == 0, 'An error occurred during interpolation'

    return values


_myEvaluateD = libNDTable.evaluate_derivative
_myEvaluateD.argtypes = [POINTER(NDTable_t), c_void_p, c_void_p, c_int,
                            c_int, c_int, c_int, c_void_p]
_myEvaluateD.restype = c_int

def _derivate(data, points, deltas, interp, extrap):
    values = np.empty(points[0].shape)
    # params
    params = (c_void_p * len(points))()
    delta_params = (c_void_p * len(points))()

    for i, param in enumerate(points):
        params[i] = param.ctypes.get_as_parameter()
    for i, delta in enumerate(deltas):
        delta_params[i] = delta.ctypes.get_as_parameter()


    res = _myEvaluateD(byref(NDTable_t(data=data)),
                          params,
                          c_int(len(params)),
                          c_int(INTERP_METH[interp]),
                          c_int(EXTRAP_METH[extrap]),
                          c_int(values.size),
                          values.ctypes.get_as_parameter(),
                          delta_params
                          )
    assert res == 0, 'An error occurred during interpolation'

    return values

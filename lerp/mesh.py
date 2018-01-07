# -*- coding: utf-8 -*-
"""
This module delivers utilities to manipulate data meshes

"""

import numpy as np
from lerp.intern import logger, myPlot
from lerp.core import path2so
from lerp.core.config import get_option

from functools import (partial, wraps)
import pickle
from xarray import DataArray

import xml.etree.ElementTree as ET

from numpy.ctypeslib import (ndpointer, load_library)
from ctypes import (c_void_p, c_int, c_double, cdll, byref, POINTER, Structure)
import ctypes
from enum import IntEnum

import sys
from os.path import (dirname, join as pjoin)
from copy import (copy, deepcopy)


# Base class for creating enumerated constants that are
# also subclasses of int
#
# http://www.chriskrycho.com/2015/ctypes-structures-and-dll-exports.html
# Option 1: set the _as_parameter value at construction.
# def __init__(self, value):
#    self._as_parameter = int(value)
#
# Option 2: define the class method `from_param`.
# @classmethod
# def from_param(cls, obj):
#    return int(obj)
class LookUpEnum(IntEnum):
    @classmethod
    def from_param(cls, obj):
        return int(obj)

INTERP_METH = LookUpEnum('INTERP_METH',
                         'hold nearest linear akima fritsch_butland steffen')
EXTRAP_METH = LookUpEnum('EXTRAP_METH',
                         'hold linear')



libNDTable = load_library('libNDTable', path2so)

MAX_NDIMS = 32
ARRAY_MAX_NDIMS = c_int * MAX_NDIMS
POINTER_TO_DOUBLE = ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
POINTER_TO_BP = POINTER_TO_DOUBLE * MAX_NDIMS

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
                ("breakpoints", POINTER_TO_BP)]

    def __init__(self, *args, **kwargs):
        if 'data' in kwargs:
            _mesh = kwargs['data']
            data = _mesh.data.astype(np.float64)
            kwargs['data'] = data.ctypes.data_as(POINTER_TO_DOUBLE)
            kwargs['shape'] = ARRAY_MAX_NDIMS(*data.shape)
            kwargs['strides'] = ARRAY_MAX_NDIMS(*data.strides)
            kwargs['itemsize'] = data.itemsize
            kwargs['ndim'] = data.ndim
            kwargs['size'] = data.size
            kwargs['breakpoints'] = POINTER_TO_BP(*[np.asanyarray(getattr(_mesh, elt),
                                 dtype=np.float64, order='C').ctypes.data
                           for elt in _mesh.dims])

        super(NDTable_t, self).__init__(*args, **kwargs)

    @classmethod
    def from_param(cls, obj):
        return byref(obj)


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
# Import evaluate_struct
################################################################################
evaluate_struct = libNDTable.evaluate_struct
evaluate_struct.argtypes = [NDTable_t, c_void_p, c_int,
                            c_int, c_int, c_int, c_void_p]
evaluate_struct.restype = c_int

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


_html_style = {
    'table': 'border: 0px none;',
    'th': 'color:LightGrey;border:0px none;'
          'text-align:center;background:none;',
    'tr': 'border:0px none; border-bottom:1px solid #C0C0C0;background:none;',
    'none': 'border:0px none;background:none;',
}


def _axisnormalize(func):
    """Insure axis conversion to integer
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from inspect import signature
        sig = signature(func)

        axis = kwargs["axis"] if "axis" in kwargs \
            else sig.parameters['axis'].default

        if isinstance(axis, str):
            axis = "xyzvw".index(axis)

        ndim = self.ndim
        assert ndim > axis, \
            ValueError(f"axis ({axis}) must be in [{','.join(self.axes)}]")

        assert type(axis) is int, TypeError("Error converting axis to integer.")
        kwargs["axis"] = axis
        out = func(self, *args, **kwargs)
        return out

    return wrapper

def _StyledSubElement(parent, child):
    return ET.SubElement(parent, child,
                         {'style': _html_style[child]})

ET.StyledSubElement = _StyledSubElement



class Mesh(DataArray):
    """
    # Code example

    from lerp.mesh import Mesh

    np.random.seed(123)
    m3d = Mesh(x=[1, 2, 3, 6], y=[13, 454, 645, 1233, 1535],
               data=np.random.randn(4, 5),
               label="le label")


   with plt.style.context('ggplot'):
        plt.figure(figsize=(16,9))
        m3d.plot()
        plt.graphpaper(200, 1)

    """
    AXES = 'xyzvw'
    def __init__(self, *pargs, **kwargs):

        self._options = {
            "extrapolate": True,
            "step": False,
            "deepcopy": False
        }

        if 'coords' in kwargs:
            assert not bool(set(kwargs) & set(kwargs['coords'])), \
                "Redundant arguments in coords and kwargs"

        self.label = kwargs.pop('label') if 'label' in kwargs else None
        self.unit = kwargs.pop('unit') if 'unit' in kwargs else None

        # Intern to DataArray
        # See https://github.com/pydata/xarray/blob/master/xarray/core/dataarray.py
        if 'fastpath' not in kwargs:
            if 'coords' not in kwargs:
                kwargs['coords']= {}

            pargs = list(pargs)
            if 'data' not in kwargs:
                kwargs['data'] = pargs.pop()

            for _k, _v in zip(self.AXES, pargs):
                kwargs['coords'][_k] = _v
                pargs = []

            dims = set(self.AXES) & set(kwargs)

            if dims:
                for d in sorted(dims, key=lambda x : self.AXES.index(x)):
                    kwargs['coords'][d] = kwargs.pop(d)

            kwargs['dims'] = tuple(kwargs['coords'].keys())

        super(Mesh, self).__init__(*pargs, **kwargs)

    @property
    def options(self):
        from lerp.util import DictWrapper
        return DictWrapper(self._options)

    def __call__(self, *pargs, **kwargs):
        """
        Interpolate the function.

        Parameters
        ----------
        x  : 1D array
            x-coordinates of the mesh on which to interpolate.
        y : 1D array
            y-coordinates of the mesh on which to interpolate.

        Returns
        -------
            2D array with shape (len(x), len(y))
            The interpolated values.
        """
        if self.options.step:
            kwargs.pop('interp', None)
            kwargs.pop('extrap', None)
            return self.interpolation(interp="hold", extrap='hold',
                                      *pargs, **kwargs)
        else:
            if self.options.extrapolate and 'extrap' not in kwargs:
                kwargs.pop('extrap', None)
                return self.interpolation(extrap='linear', *pargs, **kwargs)
            else:
                return self.interpolation(*pargs, **kwargs)

    def interpolation(self, *points, interp='linear', extrap='hold', **kwargs):
        """Interpolation
        """

        AXES = self.AXES[:self.ndim]
        AXES = set(self.dims) & set(self.AXES)

        assert len(set(AXES) & set(kwargs)) + len(points) == self.ndim, \
            "Not enough dimensions for interpolation"

        # First:
        #   - convert points (tuple) to list,
        #   - clean-up arguments in case: mix usage points/kwargs
        #   - create a clean argument dict
        points = list(points)
        args = {_x : kwargs[_x] if _x in kwargs else points.pop(0)
                for _x in AXES}

        # Compute args dimensions and check compatibility without
        # broadcasting rules.
        dims = np.array([len(args[_k]) if "__len__" in dir(args[_k])
                         else 1 for _k in args])
        assert all((dims == max(dims)) + (dims == 1)), "problème"

        #
        # print([_x for _x in AXES])

        # B = np.zeros((100,1), dtype=desc)
        #args =
        _s = max(dims)

        #print([np.broadcast_to(args[_x], #(_s,)).astype(np.float64).ctypes.get_as_parameter()
        #       for _x in self.dims])


        args = [np.asarray(args[_x], np.float64)
                if "__len__" in dir(args[_x])
                else np.ones((max(dims),), np.float64) * args[_x]
                for _x in self.dims]

        # print([np.broadcast_to(np.ravel([args[_x]]), (_s,))
        #        for _x in self.dims])
        # [np.asarray(args[_x], np.float64)
        #         if "__len__" in dir(args[_x])
        #         else np.ones((max(dims),), np.float64) * args[_x]
        #         for _x in self.dims]

        values = np.empty(args[0].shape)

        paramsType = c_void_p * len(AXES)

        res = evaluate_struct(NDTable_t(data=self),
                              paramsType(*[_a.ctypes.get_as_parameter() for _a in args]),
                              c_int(self.ndim),
                              INTERP_METH[interp],
                              EXTRAP_METH[extrap],
                              c_int(values.size),
                              values.ctypes.get_as_parameter()
                              )
        assert res == 0, 'An error occurred during interpolation'

        return values[0] if len(values) == 1 else values


    def resample(self, *points, interp='linear', extrap='hold', **kwargs):
        from itertools import zip_longest
        AXES = self.AXES[:self.ndim]
        AXES = set(self.dims) & set(self.AXES)
        AXES = sorted(AXES)

        assert len(points) <= self.ndim, \
            "Too much points provided for resampling"

        assert points==() or kwargs=={}, "problème"

        if kwargs=={}:
            args = {}
            for _i, _a in enumerate(AXES):
                try:
                    args[_a] =  points[_i] if "__len__" in dir(points[_i]) else [points[_i]]
                except IndexError:
                    args[_a] =  self.coords[_a].data
        else:
            args = {_x : kwargs[_x] if _x in kwargs else self.coords[_x].data
                for _x in AXES}

        n_points = np.prod([len(args[k]) for k in args])

        print(n_points)

#        for _a in
        # [ np.repeat(args[k], np.prod([len(args[k]) for k in AXES[]]))]
        return args



        _dim = np.array([len(args[_k]) if "__len__" in dir(args[_k])
                         else 1 for _k in args])

        assert all((_dim == max(_dim)) + (_dim == 1)), "problème"

        args = [np.asarray(args[_x], np.float64)
                if "__len__" in dir(args[_x])
                else np.ones((max(_dim),), np.float64) * args[_x]
                for _x in AXES]

        values = np.empty(args[0].shape)

        # params
        params = (c_void_p * len(args))()
        for i, param in enumerate(args):
            params[i] = param.ctypes.get_as_parameter()

        res = evaluate_struct(byref(NDTable_t(data=self)),
                              params,
                              c_int(len(params)),
                              c_int(INTERP_METH[interp]),
                              c_int(EXTRAP_METH[extrap]),
                              c_int(values.size),
                              values.ctypes.get_as_parameter()
                              )
        assert res == 0, 'An error occurred during interpolation'

        return values[0] if len(values) == 1 else values

    # Plot MAP as PDF in filename
    def plot(self, xy=False, filename=None, **kwargs):

        import matplotlib.pyplot as plt

        assert self.ndim <= 2, "More that two dimensions"

        if self.label is None:
            self.label = ""
        if self.unit is None:
            self.unit = ""

        x_axis = self.coords[self.dims[0]]
        y_axis = self.coords[self.dims[1]] if self.ndim > 1 else None

        plt.xlabel(f"{x_axis.label} [{x_axis.unit}]"
                   if x_axis.label is not None else "Label []")
        plt.ylabel(self.label + ' [' + self.unit + ']')

        if y_axis is not None:
            for _i, _y in enumerate(y_axis.data):
                # print("plot {}".format(_x))
                plt.plot(x_axis.data,
                         self.data.take(_i, axis=1),
                         '-', linewidth=1, label=f"{_y} {y_axis.unit}",
                         **kwargs)
#                         self.data.take(_i, axis=self.AXES.index(self.dims[1])),

        else:
            plt.plot(x_axis.data, self.data, '-', linewidth=1,
                     label=f"{x_axis.unit}", **kwargs)

        plt.legend(loc=2, borderaxespad=0., frameon=0)

        if filename is not None:
            print("Save file as " + filename)
            plt.savefig(filename, bbox_inches='tight')


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

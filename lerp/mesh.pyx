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
from ctypes import (c_void_p, c_int, cdll, byref, POINTER, Structure)
from enum import IntEnum

import sys
from os.path import (dirname, join as pjoin)
from copy import (copy, deepcopy)


class LookUpEnum(IntEnum):
    @property
    def ctype(self):
        return c_int(self.value)

INTERP_METH = LookUpEnum('INTERP_METH',
                         'hold nearest linear akima fritsch_butland steffen')
EXTRAP_METH = LookUpEnum('EXTRAP_METH',
                         'hold linear')

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

        #   Si les axes manquant pas dans AXES:
        #       erreur
        #   Sinon:
        #       mappage axes -> valeurs
        # Sinon:
        #       mappage axes -> valeurs

        #if len(points) < self.ndim:


        AXES = self.AXES[:self.ndim]
        assert len(set(AXES) & set(kwargs)) + len(points) == self.ndim, \
            "Not enough dimensions for interpolation"

        points = list(points)
        args = {_x : kwargs[_x] if _x in kwargs else points.pop(0)
                for _x in AXES}

        _dim = np.array([len(args[_k]) if "__len__" in dir(args[_k])
                         else 1 for _k in args])

        assert all((_dim == max(_dim)) + (_dim == 1)), "problème"

        args = [np.asarray(args[_x], np.float64)
                if "__len__" in dir(args[_x])
                else np.ones((max(_dim),), np.float64) * args[_x]
                for _x in AXES]


        # from itertools import zip_longest
        # desc = np.dtype({'names' : AXES,
        #                  'formats' : [np.float64] * self.ndim})
        #
        # A = np.array(list(zip_longest(*points)),
        #              dtype=desc)
        # return A
        #
        # for _k, _v in zip(AXES, points):
        #     assert _k not in kwargs.keys(), f"Key {_k} already in kwargs"
        #     kwargs[_k] = _v
        # print(kwargs.fromkeys(AXES))
        # #if kwargs.keys() & AXES:

        return _interpol(self, args,
                         interp=interp, extrap=extrap)

    # Plot MAP as PDF in filename
    def plot(self, xy=False, filename=None, **kwargs):

        import matplotlib.pyplot as plt

        if self.label is None:
            self.label = ""
        if self.unit is None:
            self.unit = ""

        plt.xlabel(f"{self.y.label} [{self.y.unit}]"
                   if self.y.label is not None else "Label []")
        plt.ylabel(self.label + ' [' + self.unit + ']')

        for _i, _x in enumerate(self.x.data):
            # print("plot {}".format(_x))
            plt.plot(self.y, self.data[_i], '-', linewidth=1,
                          label=u"{} {}".format(_x, self.x.unit), **kwargs)

        plt.legend(loc=2, borderaxespad=0., frameon=0)

        if filename is not None:
            print("Save file as " + filename)
            plt.savefig(filename, bbox_inches='tight')

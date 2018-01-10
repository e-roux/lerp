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

_html_style = {
    'table': 'border: 0px none;',
    'th': 'color:LightGrey;border:0px none;'
          'text-align:center;background:none;',
    'tr': 'border:0px none; border-bottom:1px solid #C0C0C0;background:none;',
    'none': 'border:0px none;background:none;',
}


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
            kwargs['breakpoints'] = POINTER_TO_BP(*[np.asanyarray(_mesh.coords[elt],
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
# Import evaluate_interpolation
################################################################################
evaluate_interpolation = libNDTable.evaluate_interpolation
evaluate_interpolation.argtypes = [NDTable_t, c_void_p, c_int,
                            c_int, c_int, c_int, c_void_p]
evaluate_interpolation.restype = c_int

evaluate_derivative = libNDTable.evaluate_derivative
evaluate_derivative.argtypes = [NDTable_t, c_void_p, c_void_p, c_int,
                            c_int, c_int, c_int, c_void_p]
evaluate_derivative.restype = c_int


class BreakPoints(np.ndarray):
    """
    Basic subclass of `numpy.ndarray` with `unit` and `label` attributes.
    Parameters
    ----------
    d : array
           Data in form of a python array or numpy.array
    label : string, optional
            label for plotting utility
    unit : string, optional
           unit for plotting utility
    Examples
    --------
    .. code-block:: python
        In [1]: from lerp import mesh1d
        In [2]: x = mesh1d(d=[100, 600, -200, 300], label="Current", unit="A")
        In [3]: print("Max : {0} {1.unit}".format(x.max(), x))
        Max : 600 A
        In [4]: print("Before sorting\\n", x)
        Before sorting
         [ 100  600 -200  300]
        In [5]: x.sort()
        In [6]: print("After sorting\\n", x)
        After sorting
         [-200  100  300  600]
        # Element added to x, inplace and sorting
         In [7]: x.push( [250, 400 ], sort=True)
        Out[7]: mesh1d(d=[-200,100,250,300,400,600], label=Current, unit=A)
        # Addition, in place
        In [8]: x += 900
        In [9]: x
        Out[9]: mesh1d(d=[700,1000,1200,1500], label=Current, unit=A)
        # Slicing
        In [10]: x[2]
        Out[10]: 1200
        In [11]: x[1:3]
        Out[11]: mesh1d(d=[1000,1200], label="Current", unit="A")
    """

    def __new__(cls, d=[], label=None, unit=None):
        from functools import singledispatch

        # We first cast to be our class type
        # np.asfarray([], dtype='float64')
        @singledispatch
        def myArray(o):
            # if o is None:
            #    o = []
            # Will call directly __array_finalize__
            obj = np.asarray(o).ravel().view(cls)
            obj.label = label
            obj.unit = unit
            return obj

        @myArray.register(BreakPoints)
        def _(o):
            # Override label and unit if given as parameter
            if label:
                o.label = label
            if unit:
                o.unit = unit
            return o

        return myArray(d)

    def __array_finalize__(self, obj):
        # https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
        self.unit = getattr(obj, 'unit', None)
        self.label = getattr(obj, 'label', None)

    def __repr__(self):
        return '{0}, label="{1.label}", unit="{1.unit}")'. \
            format(np.array_repr(self, precision=2).replace(")", "").
                   replace("(", "(d="), self)

    def _repr_html_(self):
        max_rows = get_option("display.max_rows")

        root = ET.Element('div')
        pre = ET.SubElement(root, 'p')
        ET.SubElement(pre, 'code').text = self.__class__.__name__
        ET.SubElement(pre, 'span').text = ": "
        ET.SubElement(pre, 'b').text = self.label or "Label"
        ET.SubElement(pre, 'span').text = " [{}]".format(self.unit or "unit")
        ET.SubElement(pre, 'br')

        res = ET.SubElement(pre, 'p')
        if self.size == 1:
            res.text = str(self)
        else:
            table = ET.StyledSubElement(res, 'table')
            tbody = ET.SubElement(table, 'tbody')
            for _i in range(2):
                if not _i:
                    tr = ET.StyledSubElement(tbody, 'tr')
                    for _node in islice(np.arange(len(self)), max_rows - 1):
                        ET.StyledSubElement(tr, 'th').text = str(_node)
                    if len(self) > max_rows:
                        ET.StyledSubElement(tr, 'th').text = "..."
                        ET.StyledSubElement(tr, 'th').text = str(len(self) - 1)
                    elif len(self) > max_rows - 1:
                        ET.StyledSubElement(tr, 'th').text = str(len(self) - 1)
                else:
                    tr = ET.SubElement(tbody, 'tr',
                                       {'style': 'border: 0px solid'})
                    for _node in islice(self, max_rows - 1):
                        ET.SubElement(tr, 'td').text = str(_node)
                    if len(self) > max_rows:
                        ET.SubElement(tr, 'td').text = "..."
                        ET.SubElement(tr, 'td').text = str(self[-1])
                    elif len(self) > max_rows - 1:
                        ET.SubElement(tr, 'td').text = str(self[-1])

        return str(ET.tostring(root, encoding='utf-8'), 'utf-8')

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

        from xarray.core.variable import (Variable, as_compatible_data)
        from xarray.core.dataarray import _infer_coords_and_dims

        coords, variable, name = (None, None, None)

        self._options = {
            "extrapolate": True,
            "step": False,
            "deepcopy": False
        }

        if 'coords' in kwargs:
            assert not bool(set(kwargs) & set(kwargs['coords'])), \
                "Redundant arguments in coords and kwargs"

        for info in ["label", "unit"]:
            setattr(self, info, kwargs[info] if info in kwargs else None)

        # print(kwargs["attrs"] if "attrs" in kwargs else None) # = {"x" :"aze"}

        # Intern to DataArray
        # See
        # https://github.com/pydata/xarray/blob/master/xarray/core/dataarray.py
        if 'fastpath' not in kwargs:
            if 'coords' not in kwargs:
                coords = {}

            # Set the main data
            if 'data' not in kwargs:
                *pargs, data = pargs
            else:
                data = kwargs.pop('data')

            for _k, _v in zip(self.AXES, pargs):
                coords[_k] = _v
                pargs = []

            dims = set(self.AXES) & set(kwargs)

            if dims:
                for d in sorted(dims, key=lambda x : self.AXES.index(x)):
                    coords[d] = kwargs.pop(d)

            dims = tuple(coords.keys())
            encoding = getattr(data, 'encoding', None)
            # attrs = getattr(data, 'attrs', None)
            name = getattr(data, 'name', None)
            attrs = {"x" :"aze"}

            data = as_compatible_data(data)
            coords, dims = _infer_coords_and_dims(data.shape, coords, dims)
            variable = Variable(dims, data, attrs, encoding, fastpath=True)

        else:
            variable = pargs[0]
            #print("fastpath", *pargs, end="END\n")
            self.label = "Le label des x!"
            self.unit = "%"

        # These fully describe a DataArray
        self._variable = variable
        self._coords = coords
        self._name = name

        self._file_obj = None

        self._initialized = True

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

        args = [np.asarray(args[_x], np.float64)
                if "__len__" in dir(args[_x])
                else np.ones((max(dims),), np.float64) * args[_x]
                for _x in self.dims]

        values = np.empty(args[0].shape)

        c_params_p = c_void_p * len(self.dims)

        res = evaluate_interpolation(NDTable_t(data=self),
                            c_params_p(*[_a.ctypes.get_as_parameter()
                                           for _a in args]),
                              c_int(self.ndim),
                              INTERP_METH[interp],
                              EXTRAP_METH[extrap],
                              c_int(values.size),
                              values.ctypes.get_as_parameter()
                              )

        assert res == 0, 'An error occurred during interpolation'

        return values[0] if len(values) == 1 else values


    def derivate(self, *points, interp='linear', extrap='hold', n=1, **kwargs):
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

        args = [np.asarray(args[_x], np.float64)
                if "__len__" in dir(args[_x])
                else np.ones((max(dims),), np.float64) * args[_x]
                for _x in self.dims]

        dxi = [np.ones_like(_x) for _x in args]

        values = np.empty(args[0].shape)

        c_params_p = c_void_p * len(self.dims)

        res = evaluate_derivative(NDTable_t(data=self),
                              c_params_p(*[_a.ctypes.get_as_parameter()
                                           for _a in args]),
                              c_params_p(*[_a.ctypes.get_as_parameter()
                                           for _a in dxi]),
                              c_int(self.ndim),
                              INTERP_METH[interp],
                              EXTRAP_METH[extrap],
                              c_int(values.size),
                              values.ctypes.get_as_parameter()
                              )

        assert res == 0, 'An error occurred during interpolation'

        return values[0] if len(values) == 1 else values

    def resample(self, *points, interp='linear', extrap='hold', **kwargs):

        # First:
        #   - convert points (tuple) to list,
        #   - clean-up arguments in case: mix usage points/kwargs
        #   - create a clean argument dict

        points = list(points)
        args = {}
        for d in self.dims:
            if d in kwargs:
                args[d] = kwargs[d]
            else:
                try:
                    args[d] = points.pop(0)
                except IndexError:
                    args[d] = self.coords[d]

        mg = np.meshgrid(*args.values(), indexing='ij')
        #return args
        nv = self.interpolation(*mg, interp=interp, extrap=extrap)
        return Mesh(nv, **args)

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

        else:
            plt.plot(x_axis.data, self.data, '-', linewidth=1,
                     label=f"{x_axis.unit}", **kwargs)

        plt.legend(loc=2, borderaxespad=0., frameon=0)

        if filename is not None:
            print("Save file as " + filename)
            plt.savefig(filename, bbox_inches='tight')

    # def _repr_html_(self):
    #     max_rows = get_option("display.max_rows")
    #
    #     root = ET.Element('div')
    #     pre = ET.SubElement(root, 'p')
    #     ET.SubElement(pre, 'code').text = f"{self.__class__.__name__ }: "
    #     ET.SubElement(pre, 'b').text = self.label or "Label"
    #     ET.SubElement(pre, 'span').text = " [{}]".format(self.unit or "unit")
    #     ET.SubElement(pre, 'br')
    #
    #     for _a in self.dims:
    #         axis = self.coords[_a]
    #         #root.append(ET.fromstring(axis._repr_html_()))
    #
    #     return str(ET.tostring(root, encoding='utf-8'), 'utf-8')
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

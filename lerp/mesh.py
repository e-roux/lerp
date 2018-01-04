# -*- coding: utf-8 -*-
"""
This module delivers utilities to manipulate data meshes

"""
import os.path
import pickle
from collections import namedtuple
from copy import (copy, deepcopy)
from itertools import islice, product
from numbers import Number
import numpy as np

from numpy.core.multiarray import interp as interp2d
from scipy.interpolate import dfitpack, fitpack
import xml.etree.ElementTree as ET
from lerp.intern import logger, myPlot
from lerp.core.config import get_option
from functools import (partial, wraps)
import inspect
from os import path
import sys
from ctypes import c_void_p, c_int, cdll
from numpy.ctypeslib import ndpointer
import abc
from lerp.core.interpolation import *
from xarray import DataArray


axis = namedtuple('axis', ['label', 'unit'])

axeConv = {_i: _j for (_i, _j) in enumerate('xyzvw')}

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


class Mesh_old(object):
    """Docstring.

    Parameters
    ----------
    """

    def __new__(cls, *pargs, **kwargs):

        """
        pargs peut contenir des vecteurs "breakpoints"
        ils peuvent être renseignés également en passant des
        arguments nommés dans kwargs
        """
        cls.axes = set("xyzvw") & set(kwargs)
        cls.ndim = len(cls.axes)

        cls._options = {
            "extrapolate": True,
            "step": False,
            "deepcopy": False
        }

        def get_value(self, axis=None):
            # if "_NDTable" in self.__dict__:
            #    del self._NDTable
            return getattr(self, f"_{axis}")

        def set_value(self, value, axis=None):
            obj = getattr(self, f"_{axis}")
            setattr(self, f"_{axis}",
                    obj.__class__(value, **obj.__dict__))

        def setd(self, obj):
            self._d = obj

        cls.d = property(fget=partial(get_value, axis="d"),
                         fset=setd)

        for axe in cls.axes:
            print(f"Got : {axe} -> {kwargs[axe]}")
            setattr(cls, f"_{axe}", BreakPoints(kwargs[axe]))
            setattr(cls, axe,
                    property(fget=partial(get_value, axis=axe),
                             fset=partial(set_value, axis=axe)))

        # dynamicaly write special methods

        setattr(cls, "__neg__",
                lambda self : self.__class__(**{ax:getattr(self, f"_{ax}")
                                                for ax in cls.axes},
                                             d=-self.d, label=cls.label,
                                             unit=self.unit))

        for method in ["argmax", "argmin", "unique", "min",
                       "max", "mean", "median"]:
            setattr(cls, method,
                lambda cls, *args, **kwargs : getattr(np, method)(cls._d, *args, **kwargs))

        if cls.axes:
            if "_d" not in dir(cls):
                cls._d = np.zeros([len(getattr(cls, f"_{axe}")) for axe in cls.axes])

        return cls

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

    @property
    def options(self):
        from lerp.util import DictWrapper
        return DictWrapper(self._options)

    def __dir__(self):
        return sorted([f for f in dir(self.__class__)
                       if not (f.startswith('_') |
                               ('deprecated' in repr(getattr(self, f))))],
                      key=lambda x: x.lower())

    def __eq__(self, other):

        if all((np.all(getattr(self, _axis) == getattr(other, _axis))
                for _axis in self.axes)):
            if self.label != other.label:
                print(f"Labels are differents : {self.label} / {other.label}")
            if self.unit != other.unit:
                print("Units are differents : {self.unit} / {other.unit}")
            return True
        else:
            return False

    def __len__(self):
        return self._d.size

    def __sub__(self, obj):
        return self.__add__(-obj)

    def _repr_html_(self):
        max_rows = get_option("display.max_rows")

        root = ET.Element('div')
        pre = ET.SubElement(root, 'p')
        ET.SubElement(pre, 'code').text = f"{self.__class__.__name__ }: "
        ET.SubElement(pre, 'b').text = self.label or "Label"
        ET.SubElement(pre, 'span').text = " [{}]".format(self.unit or "unit")
        ET.SubElement(pre, 'br')

        for _a in self.axes:
            axis = getattr(self, _a)
            root.append(ET.fromstring(axis._repr_html_()))

        return str(ET.tostring(root, encoding='utf-8'), 'utf-8')

    def apply(self, func, axis="d", inplace=False):
        """Apply a function along axis

        Parameters
        ----------
        func :   function
        axis:    string
        inplace: boolean
                 True for inplcae mofication

        Returns
        -------
        obj if inplace is True

        """
        axislist = set(self.__dict__.keys()) & {f"_{a}" for a in "xyzvwd"}
        assert f"_{axis}" in axislist, ValueError(f"{axis} must be ",
                                            ", ".join(axislist))

        obj = self if inplace else copy(self)

        # Apply function
        setattr(obj, axis,
                np.apply_along_axis(func, 0, getattr(self, axis))
                )

        if not inplace:
            return obj

    def copy(self):
        return deepcopy(self) if self.options.deepcopy else copy(self)

    @_axisnormalize
    def diff(self, axis=0, n=1):
        """
        """
        out = copy(self)
        setattr(out, "xyzvw"[axis],
                getattr(out, "xyzvw"[axis])[:-n])
        out.d = np.diff(out.d, n=n, axis=axis)
        return out

    def dropnan(self):
        """Drop NaN values and return new mesh2d."""
        axislist = set(self.__dict__.keys()) & {f"_{a}" for a in "xyzvw"}
        return self[~np.isnan(self.x)]

    @property
    def shape(self):
        """Get object shape."""
        mylen = (len(getattr(self, _a)) for _a in self.axes)
        return tuple(mylen)

    def read_pickle(self, fileName=None):
        """Read from pickle."""
        try:
            fileName = os.path.normpath(fileName)
            os.path.exists(fileName)
            with open(fileName, 'rb') as f:
                # The protocol version used is detected automatically,
                # so we do not have to specify it.
                data = pickle.load(f)
            return data
        except OSError:
            raise FileNotFoundError(f"Please check your path, "
                                    f"{fileName} not found.")

    def reshape(self, sort=True):
        """
        """
        assert all(self.shape), \
            ValueError(f"One axis has at least dim zero")

        self.d = np.reshape(self.d, self.shape)
        self.sort()

    def to_pickle(self, fileName=None):
        """Simple export to pickle."""
        try:
            fileName = os.path.normpath(fileName)
            with open(fileName, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        except OSError:
            raise FileNotFoundError(f"Please check your path, "
                                    f"{fileName} not found.")

    # TODO : check dtype
    @_axisnormalize
    def push(self, s=None, d=None, axis=0, inplace=False):
        """
        # s: support point ; d = data (np.array de dimension n * 1)
        """
        _axe = getattr(self, self.axes[axis])
        w2add = np.zeros(len(_axe))
        w2add.put(np.arange(len(d)), d)

        if s not in _axe:
            at = np.searchsorted(_axe, s)
            setattr(self, self.axes[axis],
                    _axe.insert(at, s))

            if self.d.size > 0:
                self.d = np.insert(self.d, at, w2add, axis=axis)
            else:
                self.d = [np.array(w2add)]
        else:
            print("push: Value already defined at {}".format(s))

        self.reshape()


    @_axisnormalize
    def append(self, arr, values, axis=None, inplace=False):
        """
        Append values to a mesh.

        Parameters
        ----------
        arr : array_like
            Values a.
        values : array_like
            These values are appended to a copy of `arr`.  It must be of the
            correct shape (the same shape as `arr`, excluding `axis`).  If
            `axis` is not specified, `values` can be any shape and will be
            flattened before use.
        axis : int, optional
            The axis along which `values` are appended.  If `axis` is not
            given, both `arr` and `values` are flattened before use.
        Returns
        -------
        append : ndarray
            A copy of `arr` with `values` appended to `axis`.  Note that
            `append` does not occur in-place: a new array is allocated and
            filled.  If `axis` is None, `out` is a flattened array.
        """
        _axe = getattr(self, self.axes[axis])
        w2add = np.zeros(len(_axe))
        w2add.put(np.arange(len(d)), d)

        if s not in _axe:
            at = np.searchsorted(_axe, s)
            setattr(self, self.axes[axis],
                    _axe.insert(at, s))

            if self.d.size > 0:
                self.d = np.insert(self.d, at, w2add, axis=axis)
            else:
                self.d = [np.array(w2add)]
        else:
            print("push: Value already defined at {}".format(s))

        self.reshape()

    def sort(self):
        """Sort the grid, ascending values."""
        for _i, _a in enumerate(self.axes):
            order = [slice(None)] * self.ndim
            axis = getattr(self, _a)
            if not np.all(axis[1:] >= axis[:-1]):
                _o = np.argsort(axis)
                setattr(self, _a, axis[_o])
                order[_i] = _o
                self.d = self.d[order]

    @property
    def T(self):
        """Transpose mesh."""
        obj = copy(self)
        for axis, axis_t in zip(self.axes, self.axes[::-1]):
            setattr(obj, axis,
                    getattr(self, axis_t))
        obj.d = self.d.T
        return obj

    def interpolation(self, *points, interp='linear', extrap='hold'):
        """Purpose of this method is to return a linear interpolation
        of a d vector for an unknown value x. If the targeted value
        is out of the x range, the returned d-value is the first,
        resp. the last d-value.

        No interpolation is made out of the x definition range. For such
        a functionality, use:py:meth:`extrapolate` instead.

        Parameters
        ----------
        x:: iterable or single element,

        kind: str or int, optional
        Specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic', 'cubic' where 'slinear', 'quadratic'
        and 'cubic' refer to a spline interpolation of first, second or third
        order) or as an integer specifying the order of the spline
        interpolator to use. Default is 'linear'.

        Returns
        -------
        A single element or a :class:`numpy.array` if the x parameter was
        a :class:`numpy.array` or a list
        """
        return _interpol(self, *points, interp=interp, extrap=extrap)



    def evaluate_derivative(self, *points, dx=None, interp='linear', extrap='hold'):
        """Purpose of this method is to return a linear interpolation
        of a d vector for an unknown value x. If the targeted value
        is out of the x range, the returned d-value is the first,
        resp. the last d-value.

        No interpolation is made out of the x definition range. For such
        a functionality, use:py:meth:`extrapolate` instead.

        Parameters
        ----------
        x:: iterable or single element,

        kind: str or int, optional
        Specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic', 'cubic' where 'slinear', 'quadratic'
        and 'cubic' refer to a spline interpolation of first, second or third
        order) or as an integer specifying the order of the spline
        interpolator to use. Default is 'linear'.

        Returns
        -------
        A single element or a :class:`numpy.array` if the x parameter was
        a :class:`numpy.array` or a list
        """
        # convert the arguments to double arrays
        data = np.asanyarray(self.d, dtype=np.float64, order='C')
        # Convert data shape tuple to array
        shape = np.asarray(data.shape, np.int32)

        #---------------------------------------------------------
        # Values to interpolate
        points = [np.asarray(points[i], np.float64)
                  for  i, _ in enumerate(points)]
        values = np.empty(points[0].shape)

        #---------------------------------------------------------
        # params
        params = (c_void_p * len(points))()
        for i, param in enumerate(points):
            params[i] = param.ctypes.get_as_parameter()
        #---------------------------------------------------------
        # breakpoints
        breakpoints = [np.asanyarray(getattr(self, elt),
                                     dtype=np.float64, order='C')
                       for elt in self.axes]

        breakpoints_ = (ndpointer(dtype=np.float64,
                                  flags='C_CONTIGUOUS') * 32)()
        for i, scale in enumerate(breakpoints):
            breakpoints_[i] = scale.ctypes.data

        # Values to interpolate
        #dx_ = (c_void_p * len(dx))()
        #dx_[0] = dx.ctypes.get_as_parameter()


        dx_ = [np.asarray(dx[i], np.float64)
                  for  i, _ in enumerate(points)]

        # res = _myEvaluateD(
        #                   data,
        #                   c_int(data.ndim),
        #                   breakpoints_,
        #                   shape.ctypes.get_as_parameter(),
        #                   params,
        #                   c_int(len(params)),
        #                   INTERP_METH[interp].ctype,
        #                   EXTRAP_METH[extrap].ctype,
        #                   c_int(values.size),
        #                   values.ctypes.get_as_parameter(),
        #                   dx_.ctypes.get_as_parameter()
        #                  )
        # assert res == 0, 'An error occurred during interpolation'

        return values






class _mesh(abc.ABC):
    """Docstring.

    Parameters
    ----------
    """

    def __init_subclass__(cls, ndim=None, *pargs, **kwargs):

        for _cls in inspect.getmro(cls):
            if _cls.__name__ == "mesh2d":
                ndim = 2
                break
            elif _cls.__name__ == "mesh3d":
                ndim = 3
                break
            elif _cls.__name__ == "mesh4d":
                ndim = 4
                break
            elif _cls.__name__ == "mesh5d":
                ndim = 5
                break

        cls.ndim = ndim - 1
        cls.axes = "xyzvw"[:ndim-1]

        axs = "xyzvw"[:ndim-1]

        cls._options = {
            "extrapolate": True,
            "step": False,
            "deepcopy": False
        }

        def get_value(self, axis=None):
            # if "_NDTable" in self.__dict__:
            #    del self._NDTable
            return getattr(self, f"_{axis}")

        def set_value(self, value, axis=None):
            obj = getattr(self, f"_{axis}")
            setattr(self, f"_{axis}",
                    obj.__class__(value, **obj.__dict__))

        def setd(self, obj):
            self._d = obj

        cls.d = property(fget=partial(get_value, axis="d"),
                         fset=setd)

        for ax in axs:
            setattr(cls, ax,
                    property(fget=partial(get_value, axis=ax),
                             fset=partial(set_value, axis=ax)))

        # dynamicaly write special methods

        setattr(cls, "__neg__",
                lambda self : self.__class__(**{ax:getattr(self, f"_{ax}")
                                                for ax in axs},
                                             d=-self.d, label=self.label,
                                             unit=self.unit))
        for method in ["argmax", "argmin", "unique"]:
            setattr(cls, method,
                lambda self, *args, **kwargs : getattr(np, method)(self.d, *args, **kwargs))

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

    @property
    def options(self):
        from lerp.util import DictWrapper
        return DictWrapper(self._options)

    def update(self, **kwargs):
        for _arg in kwargs:
            setattr(self, _arg, kwargs[_arg])

    def __dir__(self):
        return sorted([f for f in dir(self.__class__)
                       if not (f.startswith('_') |
                               ('deprecated' in repr(getattr(self, f))))],
                      key=lambda x: x.lower())

    def __eq__(self, other):

        if all((np.all(getattr(self, _axis) == getattr(other, _axis))
                for _axis in self.axes)):
            if self.label != other.label:
                print(f"Labels are differents : {self.label} / {other.label}")
            if self.unit != other.unit:
                print("Units are differents : {self.unit} / {other.unit}")
            return True
        else:
            return False

    def __len__(self):
        return self._d.size

    def __sub__(self, obj):
        return self.__add__(-obj)

    def _repr_html_(self):
        max_rows = get_option("display.max_rows")

        root = ET.Element('div')
        pre = ET.SubElement(root, 'p')
        ET.SubElement(pre, 'code').text = f"{self.__class__.__name__ }: "
        ET.SubElement(pre, 'b').text = self.label or "Label"
        ET.SubElement(pre, 'span').text = " [{}]".format(self.unit or "unit")
        ET.SubElement(pre, 'br')

        for _a in self.axes:
            axis = getattr(self, _a)
            root.append(ET.fromstring(axis._repr_html_()))

        return str(ET.tostring(root, encoding='utf-8'), 'utf-8')

    def apply(self, func, axis="d", inplace=False):
        """Apply a function along axis

        Parameters
        ----------
        func :   function
        axis:    string
        inplace: boolean
                 True for inplcae mofication

        Returns
        -------
        obj if inplace is True

        """
        axislist = set(self.__dict__.keys()) & {f"_{a}" for a in "xyzvwd"}
        assert f"_{axis}" in axislist, ValueError(f"{axis} must be ",
                                            ", ".join(axislist))

        obj = self if inplace else copy(self)

        # Apply function
        setattr(obj, axis,
                np.apply_along_axis(func, 0, getattr(self, axis))
                )

        if not inplace:
            return obj

    def copy(self):
        return deepcopy(self) if self.options.deepcopy else copy(self)

    @_axisnormalize
    def diff(self, axis=0, n=1):
        """
        """
        out = copy(self)
        setattr(out, "xyzvw"[axis],
                getattr(out, "xyzvw"[axis])[:-n])
        out.d = np.diff(out.d, n=n, axis=axis)
        return out

    def dropnan(self):
        """Drop NaN values and return new mesh2d."""
        axislist = set(self.__dict__.keys()) & {f"_{a}" for a in "xyzvw"}
        return self[~np.isnan(self.x)]

    @property
    def shape(self):
        """Get object shape."""
        mylen = (len(getattr(self, _a)) for _a in self.axes)
        return tuple(mylen)

    def read_pickle(self, fileName=None):
        """Read from pickle."""
        try:
            fileName = os.path.normpath(fileName)
            os.path.exists(fileName)
            with open(fileName, 'rb') as f:
                # The protocol version used is detected automatically,
                # so we do not have to specify it.
                data = pickle.load(f)
            return data
        except OSError:
            raise FileNotFoundError(f"Please check your path, "
                                    f"{fileName} not found.")

    def reshape(self, sort=True):
        """
        """
        assert all(self.shape), \
            ValueError(f"One axis has at least dim zero")

        self.d = np.reshape(self.d, self.shape)
        self.sort()

    def to_pickle(self, fileName=None):
        """Simple export to pickle."""
        try:
            fileName = os.path.normpath(fileName)
            with open(fileName, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        except OSError:
            raise FileNotFoundError(f"Please check your path, "
                                    f"{fileName} not found.")

    def max(self, argwhere=False):
        """Returns the max value of the d array.

        If argwhere is set to True, returns a tuple containing
        (min(d), coordinates where d is min)

        Parameters
        ----------
        argwhere : boolean
        """

        if argwhere:
            _argmax = self.d.argmax()
            _argmax_unravel = np.unravel_index(_argmax, self.d.shape)

            _res = [getattr(self, i)[j] for (i, j) in
                    zip("xyzvw"[:self.d.ndim], _argmax_unravel)]
            _res.append(self.d.max())
            return tuple(_res)
        return max(self.d)

    def mean(self, *args, **kwargs):
        """Mean calculation."""
        return self.d.mean(*args, **kwargs)

    def median(self, *args, **kwargs):
        """Median calculation."""
        return self.d.median(*args, **kwargs)

    def min(self, argwhere=False):
        """Returns the min value of the d array.

        If argwhere is set to True, returns a tuple containing
        (min(d), coordinates where d is min)

        Parameters
        ----------
        argwhere : boolean
        """

        if argwhere:
            _argmin = self.d.argmin()
            _argmin_unravel = np.unravel_index(_argmin, self.d.shape)

            _res = [getattr(self, i)[j] for (i, j) in
                    zip("xyzvw"[:self.d.ndim], _argmin_unravel)]
            _res.append(self.d.min())
            return tuple(_res)
        return min(self.d)

    # TODO : check dtype
    @_axisnormalize
    def push(self, s=None, d=None, axis=0, inplace=False):
        """
        # s: support point ; d = data (np.array de dimension n * 1)
        """
        _axe = getattr(self, self.axes[axis])
        w2add = np.zeros(len(_axe))
        w2add.put(np.arange(len(d)), d)

        if s not in _axe:
            at = np.searchsorted(_axe, s)
            setattr(self, self.axes[axis],
                    _axe.insert(at, s))

            if self.d.size > 0:
                self.d = np.insert(self.d, at, w2add, axis=axis)
            else:
                self.d = [np.array(w2add)]
        else:
            print("push: Value already defined at {}".format(s))

        self.reshape()


    @_axisnormalize
    def append(self, arr, values, axis=None, inplace=False):
        """
        Append values to a mesh.

        Parameters
        ----------
        arr : array_like
            Values a.
        values : array_like
            These values are appended to a copy of `arr`.  It must be of the
            correct shape (the same shape as `arr`, excluding `axis`).  If
            `axis` is not specified, `values` can be any shape and will be
            flattened before use.
        axis : int, optional
            The axis along which `values` are appended.  If `axis` is not
            given, both `arr` and `values` are flattened before use.
        Returns
        -------
        append : ndarray
            A copy of `arr` with `values` appended to `axis`.  Note that
            `append` does not occur in-place: a new array is allocated and
            filled.  If `axis` is None, `out` is a flattened array.
        """
        _axe = getattr(self, self.axes[axis])
        w2add = np.zeros(len(_axe))
        w2add.put(np.arange(len(d)), d)

        if s not in _axe:
            at = np.searchsorted(_axe, s)
            setattr(self, self.axes[axis],
                    _axe.insert(at, s))

            if self.d.size > 0:
                self.d = np.insert(self.d, at, w2add, axis=axis)
            else:
                self.d = [np.array(w2add)]
        else:
            print("push: Value already defined at {}".format(s))

        self.reshape()

    def sort(self):
        """Sort the grid, ascending values."""
        for _i, _a in enumerate(self.axes):
            order = [slice(None)] * self.ndim
            axis = getattr(self, _a)
            if not np.all(axis[1:] >= axis[:-1]):
                _o = np.argsort(axis)
                setattr(self, _a, axis[_o])
                order[_i] = _o
                self.d = self.d[order]

    @property
    def T(self):
        """Transpose mesh."""
        obj = copy(self)
        for axis, axis_t in zip(self.axes, self.axes[::-1]):
            setattr(obj, axis,
                    getattr(self, axis_t))
        obj.d = self.d.T
        return obj

    def interpolation(self, *coordinates, interp='linear', extrap='hold'):
        """Purpose of this method is to return a linear interpolation
        of a d vector for an unknown value x. If the targeted value
        is out of the x range, the returned d-value is the first,
        resp. the last d-value.

        No interpolation is made out of the x definition range. For such
        a functionality, use:py:meth:`extrapolate` instead.

        Parameters
        ----------
        x:: iterable or single element,

        kind: str or int, optional
        Specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic', 'cubic' where 'slinear', 'quadratic'
        and 'cubic' refer to a spline interpolation of first, second or third
        order) or as an integer specifying the order of the spline
        interpolator to use. Default is 'linear'.

        Returns
        -------
        A single element or a :class:`numpy.array` if the x parameter was
        a :class:`numpy.array` or a list
        """
        return _interpol(self, *coordinates, interp=interp, extrap=extrap)


    def evaluate_derivative(self, *points, dx=None, interp='linear', extrap='hold'):
        """Purpose of this method is to return a linear interpolation
        of a d vector for an unknown value x. If the targeted value
        is out of the x range, the returned d-value is the first,
        resp. the last d-value.

        No interpolation is made out of the x definition range. For such
        a functionality, use:py:meth:`extrapolate` instead.

        Parameters
        ----------
        x:: iterable or single element,

        kind: str or int, optional
        Specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic', 'cubic' where 'slinear', 'quadratic'
        and 'cubic' refer to a spline interpolation of first, second or third
        order) or as an integer specifying the order of the spline
        interpolator to use. Default is 'linear'.

        Returns
        -------
        A single element or a :class:`numpy.array` if the x parameter was
        a :class:`numpy.array` or a list
        """
        points = [np.asarray(points[i], np.float64)
                  for  i, _ in enumerate(points)]
        deltas = [np.asarray(dx[i], np.float64)
                  for  i, _ in enumerate(points)]
        #---------------------------------------------------------
        return _derivate(self, points, deltas, interp, extrap)







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

        @myArray.register(mesh1d)
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

    def __eq__(self, other):
        if isinstance(other, self.__class__) and \
                np.all(np.asarray(self) == np.asarray(other)):
            if self.label != other.label:
                print(f"Labels are differents : {self.label} / {other.label}")
            if self.unit != other.unit:
                print(f"Units are differents : {self.unit} / {other.unit}")
            return True
        else:
            return False

    def apply(self, f, inplace=False):
        """
        Apply a function to the complete `mesh1d` data

        Parameters
        ----------
        f : function
        inplace: boolean
            True for modifying the `mesh1d` inplace

        Returns
        -------
        `mesh1d` if inplace is set to False
        """
        _dict = {_k: self.__dict__[_k] for _k in
                 (set(self.__dict__.keys()) -
                  set(['_x', '_y', '_z', '_v', '_w', '_d', '_steps']))}

        res = self.__class__(np.apply_along_axis(f, 0, np.array(self)),
                             **_dict)
        if not inplace:
            return res
        else:
            for i, v in enumerate(self):
                self[i] = res[i]

    def copy(self):
        return self.__class__(np.copy(self), **self.__dict__)

    def diff(self, n=1):
        return self.__class__(np.diff(self, n=n), **self.__dict__)

    def insert(self, at, obj, axis=0):
        if self.ndim == 1:
            newArray = np.insert(self, at, obj)
        else:
            newArray = self

        return newArray

    def mean(self, *args, **kwargs):
        """Mean calculation."""
        return np.mean(np.asarray(self), *args, **kwargs)

    def median(self, *args, **kwargs):
        """Median calculation."""
        return np.median(np.asarray(self), *args, **kwargs)

    @myPlot
    def plot(self, *args, **kwargs):
        return

    def pop(self, axis=0):
        """
        Pop an element of the array.

        Parameters
        ----------
        obj: class instance
            The other class must have the ``coef`` attribute.
        axis: integer
            Set to True if you want the mesh1d to be sorted (ascending)

        Returns
        -------
        mesh1d
            Please not that the element(s) are not added inplace!

        """
        return self.__class__(self[:-1], **self.__dict__), self[-1]

    def push(self, obj, unique=True, return_index=False):
        """
        Pushes an element to an array.

        Notes
        -----
        The elements are not added inplace

        Parameters
        ----------
        obj: single numeric, array, numpy.array
        sort: boolean
              True (default) if you want the mesh1d to be ascending sorted.

        Returns
        -------
        New object of the same class with new element

        Notes
        -----
        As instance of numpy array, the elements are not added inplace
        """
        # Test if self is not empty array
        _dict = {_k: self.__dict__[_k] for _k in
                 (set(self.__dict__.keys()) -
                  set(['_x', '_y', '_z', '_v', '_w', '_d', '_steps']))}
        try:
            len(self)
        except TypeError:
            res = np.array(obj)
        else:
            res = np.append(self, obj)

        if return_index is True:
            _unique = np.unique(res, return_index=True)
            return [self.__class__(_unique[0].flatten(),
                                   **_dict), _unique[1]]

        if unique is True:
            res = np.unique(res)

        return self.__class__(res, **_dict)

    def __contains__(self, item):
        return item in np.asarray(self)


mesh1d = BreakPoints


class mesh2d(_mesh, ndim=2):
    """
    Fundamental 2D object, strict monotonic

    Instantiation by giving (x, d) parameters or by loading a csv-file.

    Parameters
    ----------
    x : numpy.array or mesh1d
        1D array of x-coordinates of the mesh on which to interpolate

    d : numpy.array
        1D array of d-coordinates of the mesh on result to be interpolated
    options : dict [optiona]
    clipboard : boolean [optional]
                when set, override any instantiation with x and d

    fileName : string
        Complete address to csv-file, further

    Notes
    ----------
    Currently supported features:
        * calling the object `cur(x)` return the interpolated value at x.
        * common operations:  +, -, , /
        * standard functions:func:`len()`, :func:`print()`
    """

    def __init__(self, x=[], d=None,
                 x_label=None, x_unit=None,
                 label=None, unit=None,
                 clipboard=False, extrapolate=True,
                 contiguous=False, step=False,
                 **kwargs):

        self.options.extrapolate = extrapolate
        self.options.step = step

        if 'options' in kwargs:
            self._options = {**kwargs['options'],
                             **self.options}

        if clipboard:
            self.read_clipboard()
        elif contiguous:
            self._recArray = np.rec(x, d)
        else:
            self._x = mesh1d(x, label=x_label, unit=x_unit)
            self._d = np.array(d).ravel()

        self.sort()

        self.label = label
        self.unit = unit

    def __add__(self, obj):
        """
        Adds obj to self along d-axis

        Parameters
        ----------
        obj : Number or mesh2d like
              object to add to current object


        Returns
        ----------
        mesh2d

        Notes
        ------------
        Pay attention to 'extrapolate' options as it impacts the
        adding behavior of both arrays.

        Exemple
        ----------


        In [1]: A = mesh2d([1, 2, 3], [0.5, 6, 9.0])

        In [2]: A
        Out[2]:
        x = mesh1d(d=[1, 2, 3], label="None", unit="None")
        d = mesh1d(d=[ 0.5,  6. ,  9. ], label="None", unit="None")

        In [3]: A + 10
        Out[3]:
        x = mesh1d(d=[1, 2, 3], label="None", unit="None")
        d = mesh1d(d=[ 10.5,  16. ,  19. ], label="None", unit="None")

        In [4]: B = mesh2d([0.4, 3, 6], [0.5, 6, 9.0])

        In [5]: A + B
        Out[5]:
        x = mesh1d(d=[ 0.4,  1. ,  2. ,  3. ,  6. ],
                   label="None", unit="None")
        d = mesh1d(d=[ -2.3 ,   2.27,   9.88,  15.  ,  27.  ],
                   label="None", unit="None")

        In [6]: A.options
        Out[6]: {'extrapolate': True}

        In [7]: A.options.extrapolate = False

        In [8]: A + B
        Out[8]:
        x = mesh1d(d=[ 0.4,  1. ,  2. ,  3. ,  6. ],
                   label="None", unit="None")
        d = mesh1d(d=[  1.  ,   2.27,   9.88,  15.  ,  18.  ],
                   label="None", unit="None")

        """
        new_args = deepcopy(self.__dict__)
        if isinstance(obj, Number):
            # Casting rule from numpy
            new_args['d'] = np.add(self.d, obj)
            return self.__class__(**new_args)
        elif isinstance(obj, mesh2d):
            if self.x == obj.x:
                new_args['d'] = self.d + obj.d
            else:
                new_args['x'] = np.union1d(self.x, obj.x)
                new_args['d'] = [_y1 + _y2 for _y1, _y2
                                 in zip(self(new_args['x']),
                                        obj(new_args['x']))]
            return self.__class__(x=new_args["_x"],
                                  d=new_args['d'],
                                  label=new_args['label'],
                                  unit=new_args['unit'])
        else:
            logger.warning(f"Adding {obj.__class__.__name__} to \
            {self.__class__.__name__} failed")

    def __mul__(self, obj):
        """
        Adds obj to self along d-axis

        Parameters
        ----------
        obj : Number or mesh2d like
              object to multiply to current object


        Returns
        ----------
        mesh2d

        """
        new_args = deepcopy(self)
        if isinstance(obj, Number):
            # Casting rule from numpy
            return new_args.update(d=np.multiply(self.d, obj))
        elif isinstance(obj, mesh2d):
            new_args['x'] = np.union1d(self.x, obj.x)
            new_args['d'] = [self(_x) * obj(_x) for _x in new_args['x']]
            return self.__class__(**new_args)
        else:
            logger.warning("Multiplying {} to {} failed".format(
                obj.__class__.__name__,
                self.__class__.__name__))

    def __truediv__(self, obj):
        """
        """
        new_args = deepcopy(self.__dict__)
        if isinstance(obj, Number):
            # Casting rule from numpy
            new_args['d'] = np.divide(self.d, obj)
            return self.__class__(**new_args)
        elif isinstance(obj, mesh2d):
            new_args['x'] = np.union1d(self.x, obj.x)
            new_args['d'] = [self(_x) / obj(_x) for _x in new_args['x']]
            return self.__class__(**new_args)
        else:
            logger.warning(f"Multiplyng {obj.__class__.__name__} to "
                           f"{self.__class__.__name__} failed")

    def __getitem__(self, i=None):
        if isinstance(i, Number):
            return (self.x[i], self.d[i])
        else:
            return self.__class__(x=self.x[i], d=self.d[i])

    def __iter__(self):
        return zip(self.x, self.d)

    def __repr__(self):
        return f"x = {self.x.__repr__()}\nd = {self.d.__repr__()}"

    def _repr_html_(self):
        max_rows = get_option("display.max_rows")

        root = ET.Element('div')
        pre = ET.SubElement(root, 'p')
        ET.SubElement(pre, 'code').text = self.__class__.__name__
        ET.SubElement(pre, 'br')

        res = ET.SubElement(pre, 'p')
        table = ET.StyledSubElement(res, 'table')
        tbody = ET.SubElement(table, 'tbody')
        for _i, _v in enumerate('nxd'):
            if _i == 0:
                tr = ET.StyledSubElement(tbody, 'tr')
                ET.SubElement(tr, 'th',
                              {'style': 'border:0px none;'
                               'background:none;'})
                for _node in islice(np.arange(len(self)), max_rows - 1):
                    ET.StyledSubElement(tr, 'th').text = str(_node)
                if len(self) > max_rows:
                    ET.StyledSubElement(tr, 'th').text = "..."
                    ET.StyledSubElement(tr, 'th').text = str(len(self) - 1)
                elif len(self) > max_rows - 1:
                    ET.StyledSubElement(tr, 'th').text = str(len(self) - 1)
            else:
                tr = ET.SubElement(tbody, 'tr')
                _e = getattr(self, _v)
                label = self.x.label if _v == "x" else self.label
                unit = self.x.unit if _v == "x" else self.unit
                td = ET.SubElement(tr, 'td')
                b = ET.SubElement(td, 'b')
                b.text = label or "Label"
                span = ET.SubElement(td, 'span')
                span.text = " [{}]".format(unit or "unit")
                for _node in islice(_e, max_rows - 1):
                    ET.SubElement(tr, 'td').text = str(_node)
                if len(self) > max_rows:
                    ET.SubElement(tr, 'td').text = "..."
                    ET.SubElement(tr, 'td').text = str(_e[-1])
                elif len(self) > max_rows - 1:
                    ET.SubElement(tr, 'td').text = str(_e[-1])

        return str(ET.tostring(root, encoding='utf-8'), 'utf-8')

    def __str__(self):
        o = ""
        o += "\t".join(map(str, self.x))
        o += "\n"
        o += "\t".join(map(str, self.d))
        return o

    def push(self, x=None, d=None):
        """Pushes an element/array to the array

        Notes
        -----
        The element or the array is added and sorted inplace

        Parameters
        ----------
        x: single numeric, array, numpy.array
        d: single numeric, array, numpy.array

        """

        if x and d:
            x = np.ravel(x)
            d = np.ravel(d)
            # Indices des éléments uniquement dans x
            eltsToAdd = np.setdiff1d(x, self.x, assume_unique=True)
            try:
                # Get position of elts to be added in array
                eltsOrder = np.concatenate([np.where(x == __s)
                                            for __s in eltsToAdd],
                                           axis=1).ravel()

                x = x[eltsOrder]
                d = d[eltsOrder]

                # Unique
                [x, _o] = np.unique(x, return_index=True)


                d = d[_o]
                [self.x, indices] = self.x.push(x, return_index=True)
                # print(eltsOrder, x, d, _o, indices, self.x)
                self.d = np.concatenate((self.d, d))[indices]
            except:
                print("All elements already in orginal array")

        self.sort()

    @property
    def steps(self):
        from functools import partial

        _steps = copy(self)
        # otherise options will be shared
        _steps._options = deepcopy(self._options)

        _steps.options.step = True
        # Patch steps call
        _steps.plot = partial(self.plot, step=True)
        print(_steps.options.step)
        return _steps

    def extrapolate(self, x, *args, **kwargs):
        """np.interp function with linear extrapolation
        np.polyfit
        np.poly1d
        """
        if len(args) > 0:
            res = []
            res.append(self._extrapolate(x, **kwargs))
            for elt in args:
                res.append(self._extrapolate(elt, **kwargs))
            res = tuple(res)
        else:
            # If iterable
            if "__iter__" in dir(x):
                res = []
                for elt in x:
                    res.append(self._extrapolate(elt, **kwargs))
                res = np.array(res)
            else:
                res = self._extrapolate(x, **kwargs)
        return res

    def _extrapolate(self, x, *args, **kwargs):
        # if 'step' in kwargs and kwargs['step'] is True:
        #    return self.step(x, **kwargs)
        if x <= self.x[0]:
            res = self.d[0] + (x - self.x[0]) * \
                    (self.d[1] - self.d[0]) / (self.x[1] - self.x[0])
        elif x >= self.x[-1]:
            res = self.d[-1] + (x - self.x[-1]) * \
                    (self.d[-1] - self.d[-2]) / (self.x[-1] - self.x[-2])
        else:
            res = np.interp(x, self.x, self.d)

        return float(res)

    # Plot with matplotlib
    @myPlot
    def plot(self, *args, **kwargs):
        """
        Permet de tracer un diagramme
        """
        return

    def polyfit(self, degree=2):
        from lerp.polymesh import polymesh2d
        return polymesh2d(p=np.polyfit(self.x, self.d, degree),
                          x_label=self.x.label, x_unit=self.x.unit,
                          label=self.label, unit=self.unit)

    def read_clipboard(self):

        def _en_col(s):
            import re
            _res = re.findall(r'[\-\+\d\.]+', s)
            self.x = mesh1d([float(a) for (i, a)
                             in enumerate(_res) if not i % 2],
                            self.x.label, self.x.unit)
            self.d = mesh1d([float(a) for (i, a) in enumerate(_res) if i % 2],
                            self.label, self.unit)

        def _en_ligne(s):
            # En colonne
            s = s.split('\r\n')
            if len(s) == 2:
                self.x = mesh1d([float(a) for a in s[0].split('\t')],
                                self.x.label, self.x.unit)
                self.d = mesh1d([float(a) for a in s[1].split('\t')],
                                self.label, self.unit)

        def get_clipboard():
            # TODO : implement clipboard for linux / MacOS
            import win32clipboard
            try:
                win32clipboard.OpenClipboard()
                data = win32clipboard.GetClipboardData()
                win32clipboard.CloseClipboard()
                return data
            except:
                print("No data in the clipboard")

        s = get_clipboard().strip()

        lineBr = '\r\n'
        dataSep = '\t'

        if not s.count('.') and s.count(','):
            s = s.replace(",", '.')

        # S'il y a des points dans le presse papier,
        # on remplace les virgules par des tabulations

        # if at least one ',' and one '.'
        if s.count(',') and s.count('.'):
            s = s.replace(",", dataSep)
        elif not s.count('\t') and s.count(' '):
            s = s.replace(' ', dataSep)
        elif not s.count('\t') and s.count(';'):
            s = s.replace(';', dataSep)

        if 1 <= s.count(lineBr) <= 2:
            _en_ligne(s)
        elif s.count(lineBr) == s.count(dataSep):
            _en_col(s)
        elif (s.count(lineBr) + 1) == s.count(dataSep) \
                and not s.endswith(lineBr):
            _en_col(s)
        else:
            print("Pas importable")

    def resample(self, x):
        return self.__class__(x=mesh1d(x, **self.x.__dict__),
                              d=self(x))

    def to_clipboard(self, transpose=False, decimal=","):
        def set_clipboard(text):
            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text.encode('utf-8'),
                                            win32clipboard.CF_TEXT)
            win32clipboard.SetClipboardText(text,
                                            win32clipboard.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()

        set_clipboard('\r\n'.join([str(_x) + '\t' + str(_y) for (_x, _y)
                      in zip(self.x, self.d)]).replace('.', decimal))

    def to_csv(self, fileName=None, nbreDecimales=2):
        """
        Export CUR data into csv

        Parameters
        ----------
        fileName : String
            Complete path + filename where csv data will be wrote.
            Default to \'C:/temp/aze.csv\'

        nbreDecimales : integer
        """
        import csv

        with open(fileName, 'w') as f:
            writer = csv.writer(f, dialect=csv.excel_tab, delimiter=';')
            # writes header

            # writes channel names
            writer.writerow([self.x.label, self.label])
            # writes units
            writer.writerow([self.x.unit, self.unit])

            buf = np.vstack([self.x.transpose(), np.round(self.d.transpose(),
                                                          nbreDecimales)])

            buf = buf.transpose()
            #   Write all rows
            r, c = buf.shape
            for i in range(r):
                writer.writerow(buf[i, :])

        f.close()

    def gradient(self, x=None):
        if not hasattr(self, "_gradient"):
            _d = np.diff(self.x)
            self._gradient = \
                mesh2d(self.x, np.gradient(self.d,
                                           np.concatenate((_d, [_d[-1]]))))
        return self._gradient if x is None else self._gradient(x)

    @property
    def T(self):
        myObj = deepcopy(self)
        myObj.d, myObj.x = myObj.x, myObj.d
        myObj._sort()
        return myObj

    @property
    def difff(self):
        return np.diff(self.d) / np.diff(self.x)


class mesh3d(_mesh, ndim=3):
    """
    Interpolate over a 2-D grid.

    `x`, `y` and `d` are arrays of values used to approximate some function
    f: ``d = f(x, y)``. This class returns a function whose call method uses
    spline interpolation to find the value of new points.

    Parameters
    ----------
    x : array_like

    Y : array_like
        Arrays defining the data point coordinates.

        If the points lie on a regular grid, `x` can specify the column
        coordinates and `Y` the row coordinates

    Examples
    --------
    Construct a 2-D grid and interpolate on it:

    .. code-block:: python

        from scipy import interpolate
        x = np.arange(-5.01, 5.01, 0.25)
        y = np.arange(-5.01, 5.01, 0.25)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx**2+yy**2)

    """

    def __add__(self, other):

        if isinstance(other, mesh2d):
            if np.all(self._x == other.x):
                _d = self.d + other.d.reshape(len(other), 1)
            elif np.all(self._y == other.x):
                _d = self.d + other.d.reshape(1, len(other))
            else:
                raise TypeError("Object dimension not homogeneous in __add__.")
            return self.__class__(self._x, self._y, _d,
                                  self.label, self.unit, **self.options)
        if isinstance(other, Number):
            return self.__class__(self._x, self._y, self.d + other,
                                  self.label, self.unit, **self.options)

        X = mesh1d(np.sort(np.unique(np.concatenate((self._x, other._x)))),
                   self.x.label,
                   self.x.unit)
        Y = mesh1d(np.sort(np.unique(np.concatenate((self._y, other._y)))),
                   self._y.label,
                   self._y.unit)

        D = np.zeros((X.size, Y.size))

        for (i, x) in enumerate(X):
            for (j, y) in enumerate(Y):
                D[i][j] = self(x, y) + other(x, y)

        return self.__class__(X, Y, D, self.label, self.unit)

    def __mul__(self, obj):
        """
        """
        new_args = deepcopy(self.__dict__)
        if isinstance(obj, Number):
            # Casting rule from numpy
            new_args['d'] = np.multiply(self.d, obj)
            return self.__class__(**new_args)
        else:
            logger.warning("Multiplying {} to {} failed".format(
                obj.__class__.__name__,
                self.__class__.__name__))

    def __truediv__(self, obj):
        """
        """
        new_args = deepcopy(self.__dict__)
        if isinstance(obj, Number):
            # Casting rule from numpy
            new_args['d'] = np.divide(self.d, obj)
            return self.__class__(**new_args)
        else:
            logger.warning("Multiplyng {} to {} failed".format(
                obj.__class__.__name__,
                self.__class__.__name__))

    def __iter__(self):
        """

        :return: x[i], y[j], (i,j), w[i, j]
        """
        for (i, j), elem in np.ndenumerate(self.d):
            yield (self.x[i], self.y[j], (i, j), elem)

    def __init__(self, x=[], y=[], d=None,
                 x_label=None, x_unit=None,
                 y_label=None, y_unit=None,
                 label=None, unit=None,
                 extrapolate=True, clipboard=False,
                 sort=True, *pargs, **kwargs):
        """
        x -> row (index 0 of W)
        Y -> column (index 1 of W)
        W -> Z values

        Z
          Z
            Z
               Y   Y   Y   Y
              X
              X
              X
              X
              X

        """

        self.label = label
        self.unit = unit

        self.options.extrapolate = extrapolate

        self._x = mesh1d(x, label=x_label, unit=x_unit) \
            if "_x" not in kwargs else kwargs["_x"]

        self._y = mesh1d(y, label=y_label, unit=y_unit) \
            if "_y" not in kwargs else kwargs["_y"]

        self.d = np.empty((self.x.size, self.y.size)) \
            if d is None else np.asfarray(d, dtype='float64')

        if clipboard is True:
            self.read_clipboard()

        self.reshape(sort=sort)

    def _repr_html_(self):
        max_rows = get_option("display.max_rows")

        root = ET.Element('div')
        pre = ET.SubElement(root, 'p')
        ET.SubElement(pre, 'code').text = self.__class__.__name__ + ": "
        ET.SubElement(pre, 'b').text = self.label or "Label"
        ET.SubElement(pre, 'span').text = " [{}]".format(self.unit or "unit")
        ET.SubElement(pre, 'br')
        res = ET.SubElement(pre, 'p')
        table = ET.SubElement(res, 'table',
                              {'style': _html_style['none'],
                               'class': 'mesh3d'})
        tbody = ET.SubElement(table, 'tbody')

        for _a in np.arange(3):
            if _a == 0:
                tr = ET.SubElement(tbody, 'tr',
                                   {'style': _html_style['none']})
                ET.SubElement(tr, 'th',
                              {'colspan': '3', 'style': _html_style['none']})
                td = ET.SubElement(tr, 'th',
                                   {'colspan': str(len(self._y)),
                                    'style': _html_style['none']})
                ET.SubElement(td, 'b').text = self._y.label or "Label"
                ET.SubElement(td, 'span').text = " [{}]".format(
                    self._y.unit or "Unit")
            elif _a == 1:
                tr = ET.SubElement(tbody, 'tr',
                                   {'style': _html_style['none']})
                ET.SubElement(tr, 'th',
                              {'colspan': '3', 'style': _html_style['none']})
                for _node in islice(np.arange(len(self._y)), max_rows - 1):
                    ET.StyledSubElement(tr, 'th').text = str(_node)
                if len(self._y) > max_rows:
                    ET.StyledSubElement(tr, 'th').text = "..."
                    ET.StyledSubElement(tr, 'th').text = \
                        str(len(self._y) - 1)
                elif len(self._y) > max_rows - 1:
                    ET.StyledSubElement(tr, 'th').text = \
                        str(len(self._y) - 1)
                tr = ET.SubElement(tbody, 'tr', {'style': _html_style['none']})
                ET.SubElement(tr, 'th',
                              {'colspan': '3', 'style': _html_style['none']})
                for _node in islice(self._y, max_rows - 1):
                    ET.SubElement(tr, 'th').text = str(_node)
                if len(self._y) > max_rows:
                    ET.SubElement(tr, 'th').text = "..."
                    ET.SubElement(tr, 'th').text = str(self._y[-1])
                elif len(self._y) > max_rows - 1:
                    ET.SubElement(tr, 'th').text = str(self._y[-1])
            else:
                for _i, _v in enumerate(self._x):
                    tr = ET.SubElement(tbody, 'tr',
                                       {'style': 'border: 0px solid'})
                    if _i == 0:
                        td = ET.SubElement(tr, 'th',
                                           {'rowspan': str(len(self._x)),
                                            'style': _html_style['none']})
                        ET.SubElement(td, 'b', {'style': _html_style['none']})\
                          .text = self._x.label or "Label"
                        ET.SubElement(td, 'span',
                                      {'style': _html_style['none']})\
                          .text = " [{}]".format(self._x.unit or "Unit")

                    ET.StyledSubElement(tr, 'th').text = str(_i)
                    ET.SubElement(tr, 'th').text = str(self._x[_i])

                    for _node in islice(self.d[_i], max_rows - 1):
                        ET.SubElement(tr, 'td').text = str(_node)
                    if len(self._y) > max_rows:
                        ET.SubElement(tr, 'td').text = "..."
                        ET.SubElement(tr, 'td').text = str(self.d[_i][-1])
                    elif len(self._y) > max_rows - 1:
                        ET.SubElement(tr, 'td').text = str(self.d[_i][-1])

        return str(ET.tostring(root, encoding='utf-8'), 'utf-8')

    def __getitem__(self, sl):
        """
        """
        def _get_m2d(slx, slw):
            return mesh2d(x=slx,
                          d=mesh1d(slw, self.label, self.unit),
                          extrapolate=self.options.extrapolate)

        # Unpack slice sl in x and y compound
        slx, sly, *opts = sl if isinstance(sl, tuple) \
            else (sl, slice(None, None, None))

        try:
            isXslice = len(self.x[slx]) > 1 or False
        except TypeError:
            isXslice = False
        try:
            isYslice = len(self.y[sly]) > 1 or False
        except TypeError:
            isYslice = False

        # _get_m2d(self._y[sly], self.d[slx, sly])

        if isXslice and isYslice:
            return self.__class__(x=self._x[slx], y=self._y[sly],
                                  d=self.d[slx, sly], label=self.label,
                                  unit=self.unit,
                                  extrapolate=self.options.extrapolate)
        elif not isXslice and isYslice:
            return _get_m2d(self._y[sly], self.d[slx, sly])
        elif isXslice and not isYslice:
            return _get_m2d(self._x[slx], self.d[slx, sly])
        else:
            return (self._x[slx].flatten()[0], self._y[sly].flatten()[0],
                    self.d[slx, sly].flatten()[0])

    def __str__(self):
        """
        """
        res = "\t"
        res += "\t".join(map(str, self.y))
        if len(self.x):
            res += "\n"
        for index in range(0, len(self.x)):
            res += str(self.x[index])
            res += "\t"
            res += "\t".join(map(str, self.d[index]))
            res += "\n"
        return res

    def pop(self, axis=0):
        axisLenght = np.ma.size(self.d, axis=axis)
        if axisLenght > 1:
            W = self.d.take(-1, axis=axis)
            self.d = self.d.take(np.arange(axisLenght - 1), axis=axis)
            if axis == 0:
                self.x, popped = self.x.pop()
            else:
                self.y, popped = self.y.pop()

            return popped, W
        else:
            print("axis {} to small to pop".format({0: 'x', 1: 'y'}.get(axis)))

    def interpolate(self, x=None, y=None):
        """"""
        isxN = isinstance(x, Number)
        isyN = isinstance(y, Number)

        # x and y are numeric
        if isxN & isyN:
            if x <= self.x[0]:
                return np.interp(y, self.y, self.d[0])
            elif x >= self.x[-1]:
                return np.interp(y, self.y, self.d[-1])
            else:
                iX = np.searchsorted(self.x, x) - 1

            if y <= self.y[0]:
                return np.interp(x, self.x, self.d[:, 0])
            elif y >= self.y[-1]:
                return np.interp(x, self.x, self.d[:, -1])
            else:
                iY = np.searchsorted(self.y, y) - 1

            Z1 = self.d[iX, iY] + (self.d[iX, iY + 1] - self.d[iX, iY]) * \
                (y - self.y[iY]) / (self.y[iY + 1] - self.y[iY])
            Z2 = self.d[iX + 1, iY] + \
                (self.d[iX + 1, iY + 1] - self.d[iX + 1, iY]) * \
                (y - self.y[iY]) / (self.y[iY + 1] - self.y[iY])

            return Z1 + (Z2 - Z1) * (x - self.x[iX]) / \
                        (self.x[iX + 1] - self.x[iX])
        # x or y are numeric
        elif isxN | isyN:
            if isxN:
                y = self._y if y is None \
                    else self._y.__class__(y, **self._y.__dict__)
                return mesh2d(y, mesh1d([self.interpolate(x, _y) for _y in y],
                                        self.label, self.unit))
            else:
                x = self._x if x is None \
                    else self._x.__class__(x, **self._x.__dict__)
                return mesh2d(x, mesh1d([self.interpolate(_x, y) for _x in x],
                                        self.label, self.unit))
        # Either x nor y are numeric
        else:
            x = self._x if x is None else self._x.__class__(x,
                                                            **self._x.__dict__)
            y = self._y if y is None else self._y.__class__(y,
                                                            **self._y.__dict__)
            _, tx, _, ty, c, _, _ = dfitpack.regrid_smth(self._x, self._y,
                                                         np.ravel(self.d),
                                                         kx=1, ky=1)

            return self.__class__(x=x, y=y,
                                  d=[fitpack.bisplev(_x, _y, (tx, ty, c, 1, 1))
                                     for _x, _y in product(x, y)],
                                  label=self.label, unit=self.unit)

    def extrapolate(self, x=None, y=None):
        """"""
        isxN = isinstance(x, Number)
        isyN = isinstance(y, Number)

        # x and y are numerics
        if isxN & isyN:
            if x <= self.x[0]:
                iX = 0
            elif x >= self.x[-1]:
                iX = -2
            else:
                iX = np.searchsorted(self.x, x) - 1

            if y <= self.y[0]:
                iY = 0
            elif y >= self.y[-1]:
                iY = -2
            else:
                iY = np.searchsorted(self.y, y) - 1

            Z1 = self.d[iX, iY] + (self.d[iX, iY + 1] - self.d[iX, iY]) * \
                (y - self.y[iY]) / (self.y[iY + 1] - self.y[iY])
            Z2 = self.d[iX + 1, iY] + \
                (self.d[iX + 1, iY + 1] - self.d[iX + 1, iY]) * \
                (y - self.y[iY]) / (self.y[iY + 1] - self.y[iY])

            return Z1 + (Z2 - Z1) * \
                (x - self.x[iX]) / (self.x[iX + 1] - self.x[iX])
        # x or y is numeric
        elif isxN | isyN:
            # Save extrapolate status in options before setting to True
            extrapolate = self.options.extrapolate
            self.options.extrapolate
            if isxN:
                y = self._y if y is None \
                    else self._y.__class__(y, **self._y.__dict__)
                res = mesh2d(y, mesh1d([self(x, _y) for _y in y],
                                       self.label, self.unit))
                self.options.extrapolate = extrapolate
                return res
            else:
                x = self._x if x is None \
                    else self._x.__class__(x, **self._x.__dict__)
                res = mesh2d(x, mesh1d([self(_x, y) for _x in x],
                                       self.label, self.unit))
                self.options.extrapolate = extrapolate
                return res
                # Either x nor y is numeric
        else:
            x = self._x if x is None \
                else self._x.__class__(x, **self._x.__dict__)
            y = self._y if y is None \
                else self._y.__class__(y, **self._y.__dict__)

            return self.__class__(x=x, y=y,
                                  d=[self.extrapolate(_x, _y)
                                     for _x, _y in product(x, y)],
                                  label=self.label, unit=self.unit)

    def from_pandas(self, obj):
        self.__init__(x=obj.index.astype(float), x_label=self.x.label,
                      x_unit=self.x.unit, y=obj.columns.astype(float),
                      y_label=self.y.label, y_unit=self.y.unit,
                      d=[np.array(obj.loc[_o]) for _o in obj.index],
                      label=self.label, unit=self.unit)

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

        for _i, _x in enumerate(self.x):
            # print("plot {}".format(_x))
            self[_i].plot('-', linewidth=1,
                          label=u"{} {}".format(_x, self.x.unit), **kwargs)

        plt.legend(loc=2, borderaxespad=0., frameon=0)

        if filename is not None:
            print("Save file as " + filename)
            plt.savefig(filename, bbox_inches='tight')

    def read_clipboard(self):

        import pandas as pd
        s = pd.read_clipboard(index_col=0, decimal=",")
        self.from_pandas(s)

    def interparray(self, x, y):
        _, tx, _, ty, c, _, _ = dfitpack.regrid_smth(self._x, self._y,
                                                     np.ravel(self.d),
                                                     kx=1, ky=1)
        _sortx = np.argsort(x)
        _sorty = np.argsort(y)

        res = fitpack.bisplev(np.sort(x),
                                            np.sort(y),
                                            (tx, ty, c, 1, 1))
        return res[_sortx, _sorty]


class mesh4d(_mesh, ndim=4):
    """
    """

    def __init__(self,
                 x=[], y=[], z=[], d=None,
                 x_label=None, x_unit=None,
                 y_label=None, y_unit=None,
                 z_label=None, z_unit=None,
                 label=None, unit=None,
                 extrapolate=False, dtype='float64'):

        self.label = label
        self.unit = unit

        self.options.extrapolate = extrapolate

        self._x = mesh1d(x, label=x_label, unit=x_unit)
        self._y = mesh1d(y, label=y_label, unit=y_unit)
        self._z = mesh1d(z, label=z_label, unit=z_unit)
        self.d = np.zeros((self._x.size, self._y.size, self._z.size))\
            if d is None else np.asfarray(d, dtype=dtype)

        self.reshape()

    def __add__(self, other):
        return self.__class__(x=self._x, y=self._y, z=self._z,
                              d=self.d + other, label=self.label,
                              unit=self.unit, **self.options)

    def __getitem__(self, sl):
        """
        """
        global isXslice
        isXslice = False
        global isYslice
        isYslice = False
        global isZslice
        isZslice = False

        def _get_m2d(slx, slw):
            return mesh2d(x=slx, y=mesh1d(slw, self.label, self.unit),
                          extrapolate=self.options.extrapolate)

        def _get_m3d(slx, sly, w):
            return mesh3d(x=slx, y=sly, d=w,
                          label=self.label, unit=self.unit,
                          extrapolate=self.options.extrapolate)

        try:
            if len(sl) == 2:
                slx, sly = sl
                slz = slice(None, None, None)
            elif len(sl) == 3:
                slx, sly, slz = sl

            if isinstance(slx, slice):
                isXslice = len(self.x[slx]) > 1 or False
            if isinstance(sly, slice):
                isYslice = len(self.y[sly]) > 1 or False
            if isinstance(slz, slice):
                isZslice = len(self.z[slz]) > 1 or False

        except:
            slx, sly, slz = sl, slice(None, None, None),\
             slice(None, None, None)
            if isinstance(slx, slice):
                isXslice = len(self.x[slx]) > 1 or False
            isYslice = True
            isZslice = True

        if isXslice and isYslice and isZslice:
            return self.__class__(x=self._x[slx], y=self._y[sly],
                                  z=self._z[slz], d=self.d[slx, sly, slz],
                                  label=self.label, unit=self.unit,
                                  extrapolate=self.options.extrapolate)

        elif isXslice and isYslice and not isZslice:
            return _get_m3d(self._x[slx], self._y[sly], self.d[slx, sly, slz])
        elif isXslice and not isYslice and isZslice:
            return _get_m3d(self._x[slx], self._z[slz], self.d[slx, sly, slz])
        elif not isXslice and isYslice and isZslice:
            return _get_m3d(self._y[sly], self._z[slz], self.d[slx, sly, slz])

        elif isXslice and not isYslice and not isZslice:
            return _get_m2d(self._x[slx], self.d[slx, sly, slz])
        elif not isXslice and isYslice and not isZslice:
            return _get_m2d(self._y[sly], self.d[slx, sly, slz])
        elif not isXslice and not isYslice and isZslice:
            return _get_m2d(self._z[slz], self.d[slx, sly, slz])

        else:
            return self.d[slx, sly, slz]

    def push(self, s=None, d=None, axis=0):
        """
        # TODO : check type

        # From the axis number, get the corresponding
        # attribute x,y or z from self
        """
        wA = "_" + axeConv.get(axis)
        _axis = getattr(self, wA)
        # print(wA, _axis)

        # TODO: check that every (x,y) from mesh3d to add are the same as the
        if len(self.x) == 0:
            self.x = d.x

        if len(self.y) == 0:
            self.y = d.y

        if s not in _axis:
            at = np.searchsorted(_axis, s)
            setattr(self, wA, _axis.insert(at, s))

            if self.d.size > 0:
                if self.d.take(0, axis=axis).shape != d.d.shape:
                    raise TypeError("Shape of d ({}) not compliant with self's \
                    data shape ({})".format(self.d.take(at, axis=axis).shape,
                                            d.shape))
                if at >= np.size(self.d):
                    np.append(self.d, d.d, axis=axis)
                else:
                    self.d = np.insert(self.d, at, d.d, axis=axis)
            else:
                self.d = np.ravel(d.d)
        else:
            print("push: Value already defined at {}".format(s))

        self.reshape()

    def interpolate(self, x=None, y=None, z=None):
        """a
        """

        isxN = isinstance(x, Number)
        isyN = isinstance(y, Number)
        iszN = isinstance(z, Number)

        # x, y and z are numeric
        if isxN & isyN & iszN:
            if x <= self.x[0]:
                return self[0].interpolate(y, z)
            elif x >= self.x[-1]:
                return self[-1].interpolate(y, z)
            else:
                iX = np.searchsorted(self.x, x) - 1

            if y <= self.y[0]:
                return self[:, 0].interpolate(x, z)
            elif y >= self.y[-1]:
                return self[:, -1].interpolate(x, z)
            else:
                iY = np.searchsorted(self.y, y) - 1

            if z <= self.z[0]:
                return self[:, :, 0].interpolate(x, y)
            elif z >= self.z[-1]:
                return self[:, :, -1].interpolate(x, y)
            else:
                iZ = np.searchsorted(self.z, z) - 1

            X1 = self.x[iX]
            X2 = self.x[iX + 1]
            Y1 = self.y[iY]
            Y2 = self.y[iY + 1]
            Z1 = self.z[iZ]
            Z2 = self.z[iZ + 1]
            Z111 = self.d[iX, iY, iZ]
            Z121 = self.d[iX, iY + 1, iZ]
            Z112 = self.d[iX, iY, iZ + 1]
            Z122 = self.d[iX, iY + 1, iZ + 1]
            Z211 = self.d[iX + 1, iY, iZ]
            Z221 = self.d[iX + 1, iY + 1, iZ]
            Z212 = self.d[iX + 1, iY, iZ + 1]
            Z222 = self.d[iX + 1, iY + 1, iZ + 1]

            # Reduction along Z
            Z11 = Z111 + (Z112 - Z111) * (z - Z1) / (Z2 - Z1)
            Z12 = Z121 + (Z122 - Z121) * (z - Z1) / (Z2 - Z1)
            Z21 = Z211 + (Z212 - Z211) * (z - Z1) / (Z2 - Z1)
            Z22 = Z221 + (Z222 - Z221) * (z - Z1) / (Z2 - Z1)

            # reduction along Y
            Z1 = Z11 + (Z12 - Z11) * (y - Y1) / (Y2 - Y1)
            Z2 = Z21 + (Z22 - Z21) * (y - Y1) / (Y2 - Y1)

            return Z1 + (Z2 - Z1) * (x - X1) / (X2 - X1)
        # x or y are numeric
        # TODO : ne marche pas à partir de là !!!!

        x = x or self.x
        y = y or self.y
        z = z or self.z

        # , **self.options

        test = {
            (True, True, False): "__import__('lerp').mesh2d(\
                __import__('lerp').mesh1d(z, **self.z.__dict__),\
                __import__('lerp').mesh1d([self(x,y,_z) for _z in z],\
                label=self.label, unit=self.unit))",
            (True, False, True): "__import__('lerp').mesh2d(\
                __import__('lerp').mesh1d(y, **self.y.__dict__),\
                __import__('lerp').mesh1d([self(x,_y,z) for _y in y],\
                label=self.label, unit=self.unit))",
            (False, True, True): "__import__('lerp').mesh2d(\
                __import__('lerp').mesh1d(x, **self.x.__dict__),\
                __import__('lerp').mesh1d([self(_x,y,z) for _x in x],\
                label=self.label, unit=self.unit))",
            (True, False, False): "__import__('lerp').mesh3d(\
                __import__('lerp').mesh1d(y, **self.y.__dict__),\
                __import__('lerp').mesh1d(z, **self.z.__dict__),\
               [self(x,_y,_z) for _y, _z in  __import__('itertools').product(y,z)],\
                label=self.label, unit=self.unit)",
            (False, True, False): "__import__('lerp').mesh3d(\
                __import__('lerp').mesh1d(x, **self.x.__dict__),\
                __import__('lerp').mesh1d(z, **self.z.__dict__),\
               [self(_x,y,_z) for _x, _z in  __import__('itertools').product(x,z)],\
                label=self.label, unit=self.unit)",
            (False, False, True): "__import__('lerp').mesh3d(\
                __import__('lerp').mesh1d(x, **self.x.__dict__),\
                __import__('lerp').mesh1d(y, **self.y.__dict__),\
               [self(_x,_y,z) for _x, _y in  __import__('itertools').product(x,y)],\
                label=self.label, unit=self.unit)",
        }

        return eval(test[(isxN, isyN, iszN)],
                    {'self': self, 'x': x, 'y': y, 'z': z})


class mesh5d(_mesh, ndim=5):
    """
    """

    def __init__(self,
                 x=[], y=[], z=[], v=[], d=None,
                 x_label=None, x_unit=None,
                 y_label=None, y_unit=None,
                 z_label=None, z_unit=None,
                 v_label=None, v_unit=None,
                 label=None, unit=None,
                 extrapolate=True, dtype='float64'):

        self.label = label
        self.unit = unit

        # self.options = {}
        self.options.extrapolate = extrapolate

        self._x = mesh1d(x, label=x_label, unit=x_unit)
        self._y = mesh1d(y, label=y_label, unit=y_unit)
        self._z = mesh1d(z, label=z_label, unit=z_unit)
        self._v = mesh1d(v, label=v_label, unit=v_unit)
        self.d = np.zeros((self._x.size, self._y.size, self._z.size,
                           self._v.size)) if d is None else \
            np.asfarray(d, dtype=dtype)

        self.reshape()

    def __add__(self, other):
        return self.__class__(x=self._x, y=self._y, z=self._z,
                              v=self._v,
                              d=self.d + other, label=self.label,
                              unit=self.unit, **self.options)

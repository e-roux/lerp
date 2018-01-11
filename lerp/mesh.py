# -*- coding: utf-8 -*-
"""
This module delivers utilities to manipulate data meshes

"""

import xml.etree.ElementTree as ET
from itertools import islice

import numpy as np
from xarray import DataArray
from xarray.core.pycompat import dask_array_type
from xarray.core.formatting import (unindexed_dims_repr, dim_summary,
                                    short_dask_repr, short_array_repr, attrs_repr)
from lerp.core.config import get_option

from .core.interpolation import (interpolation, derivate)


_html_style = {
    'table': 'border: 0px none;',
    'th': 'color:LightGrey;border:0px none;'
          'text-align:center;background:none;',
    'tr': 'border:0px none; border-bottom:1px solid #C0C0C0;background:none;',
    'none': 'border:0px none;background:none;',
}

def _StyledSubElement(parent, child):
    return ET.SubElement(parent, child,
                         {'style': _html_style[child]})

ET.StyledSubElement = _StyledSubElement

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

    def __new__(cls, d=None, label=None, unit=None):
        from functools import singledispatch

        if d is None:
            d = []
        # We first cast to be our class type
        # np.asfarray([], dtype='float64')
        @singledispatch
        def my_array(o):
            # if o is None:
            #    o = []
            # Will call directly __array_finalize__
            obj = np.asarray(o).ravel().view(cls)
            obj.label = label
            obj.unit = unit
            return obj

        @my_array.register(BreakPoints)
        def _(o):
            # Override label and unit if given as parameter
            if label:
                o.label = label
            if unit:
                o.unit = unit
            return o

        return my_array(d)

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
            setattr(self, info, kwargs.pop(info) if info in kwargs else None)

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
                for d in sorted(dims, key=lambda x: self.AXES.index(x)):
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
            self.label = None
            self.unit = None

        # These fully describe a DataArray
        self._variable = variable
        self._coords = coords
        self._name = name

        self._file_obj = None

        self._initialized = True

        if 'fastpath' not in kwargs:
            pass


    def __repr__(self):
        # used for DataArray, Variable and IndexVariable
        if hasattr(self, 'name') and self.name is not None:
            name_str = '%r ' % self.name
        else:
            name_str = u''

        summary = [f'<lerp.{type(self).__name__} {name_str}({dim_summary(self)})>']


        if isinstance(getattr(self, 'variable', self)._data, dask_array_type):
            summary.append(short_dask_repr(self))
        elif self._in_memory or self.size < 1e5:
            summary.append(short_array_repr(self.values))
        else:
            summary.append(f'[{self.size} values with dtype={self.dtype}]')

        if hasattr(self, 'coords'):
            if self.coords:
                summary.append(repr(self.coords))

            unindexed_dims_str = unindexed_dims_repr(self.dims, self.coords)
            if unindexed_dims_str:
                summary.append(unindexed_dims_str)

        if self.attrs:
            summary.append(attrs_repr(self.attrs))

        return u'\n'.join(summary)

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
        return interpolation(self, *points, interp=interp, extrap=extrap
                             **kwargs)

    def derivate(self, *points, interp='linear', extrap='hold', **kwargs):
        """derivate
        """
        return derivate(self, *points, interp=interp, extrap=extrap,
                        **kwargs)

# -*- coding: utf-8 -*-
"""
This module delivers utilities to manipulate data meshes

"""

import xml.etree.ElementTree as ET
from itertools import islice

import numpy as np
from xarray.core.dataarray import DataArray
from xarray.core.pycompat import dask_array_type
from xarray.core.formatting import (unindexed_dims_repr, dim_summary,
                                    short_dask_repr, short_array_repr, attrs_repr)
from .core.config import get_option

# from .core.interpolation_ctypes import derivate

from .core.interpolation import interpolation

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

class Mesh(DataArray):
    """
    # Code example

    from lerp import Mesh

    np.random.seed(123)
    m3d = Mesh(x=[1, 2, 3, 6],
               y=[13, 454, 645, 1233, 1535],
               data=np.random.randn(4, 5),
               label="le label")


   with plt.style.context('ggplot'):
        plt.figure(figsize=(16,9))
        m3d.plot()
        plt.graphpaper(200, 1)

    """
    # AXES = 'xyzvw'
    def __init__(self, *pargs, **kwargs):

        # from xarray.core.variable import (Variable, as_compatible_data)
        # from xarray.core.dataarray import _infer_coords_and_dims

        # coords, variable, name = (None, None, None)

        self._options = {
            "extrapolate": True,
            "step": False,
            "deepcopy": False
        }

        # if 'coords' in kwargs:
        #     assert not bool(set(kwargs) & set(kwargs['coords'])), \
        #         "Redundant arguments in coords and kwargs"

        # for info in ["label", "unit"]:
        #     setattr(self, info, kwargs.pop(info) if info in kwargs else None)

        # print(kwargs["attrs"] if "attrs" in kwargs else None) # = {"x" :"aze"}

        # Intern to DataArray
        # See
        # https://github.com/pydata/xarray/blob/master/xarray/core/dataarray.py
        # if 'fastpath' not in kwargs:
        #     if 'coords' not in kwargs:
        #         coords = {}

        #     if 'data' not in kwargs:
        #         *pargs, data = pargs
        #     else:
        #         data = kwargs.pop('data')

        #     for _k, _v in zip(self.AXES, pargs):
        #         coords[_k] = _v
        #         pargs = []

        #     dims = set(self.AXES) & set(kwargs)

        #     if dims:
        #         for d in sorted(dims, key=lambda x : self.AXES.index(x)):
        #             coords[d] = kwargs.pop(d)

        #     dims = tuple(coords.keys())

        #     kwargs['coords'] = coords
        #     kwargs['data'] = data
        #     kwargs['dims'] = dims

        super(Mesh, self).__init__(*pargs, **kwargs)

        # if 'fastpath' not in kwargs:
        #     encoding = getattr(data, 'encoding', None)
        #     # attrs = getattr(data, 'attrs', None)
        #     name = getattr(data, 'name', None)
        #     attrs = {"x" :"aze"}

        #     data = as_compatible_data(data)
        #     coords, dims = _infer_coords_and_dims(data.shape, coords, dims)
        #     variable = Variable(dims, data, attrs, encoding, fastpath=True)

        # else:
        #     variable = pargs[0]
        #     #print("fastpath", *pargs, end="END\n")
        #     self.label = None
        #     self.unit = None


        # # These fully describe a DataArray
        # self._variable = variable
        # self._coords = coords
        # self._name = name

        # self._file_obj = None

        # self._initialized = True


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

    def interpolation(self, *points, interp='linear', extrap='hold'):
        """Interpolation
        """

        return interpolation(self, list(points),
                             interp=interp, extrap=extrap)

    # def derivate(self, *points, interp='linear', extrap='hold', **kwargs):
    #     """derivate
    #     """
    #     return derivate(self, *points, interp=interp, extrap=extrap,
    #                     **kwargs)

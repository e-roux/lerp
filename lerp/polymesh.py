# -*- coding: utf-8 -*-
"""
This module delivers polymesh support.

"""

import numpy as np

from lerp.mesh import mesh1d, mesh2d, mesh3d
from lerp.intern import logger
# logger


class polymesh2d(object):
    """Polynom based mash support."""

    def __init__(self, p=[], x_label=None, x_unit=None,
                 label=None, unit=None):

        self.x = axis(label=x_label, unit=x_unit)
        self.label = label
        self.unit = unit

        self.p = np.poly1d(p)

    def __call__(self, x):
        if isinstance(x, Number):
            return self.p(x)
        else:
            return mesh2d(x=mesh1d(x, self.x.label, self.x.unit), d=self.p(x),
                          label=self.label, unit=self.unit)

    def plot(self, *pargs, **kwargs):
        import matplotlib.pyplot as plt

        xlim = kwargs['xlim'] if 'xlim' in kwargs else (0, 100)
        n = kwargs.pop('n') if 'n' in kwargs else 500

        kwargs['dx'] = kwargs['dx'] if 'dx' in kwargs else None
        kwargs['ylim'] = kwargs['ylim'] if 'ylim' in kwargs else None
        kwargs['dy'] = kwargs['dy'] if 'dy' in kwargs else None

        if n < 2:
            logger.warning("More than two points are necessary for linspace")
            return

        if plt.gca().lines:
            x = np.linspace(*plt.xlim(), n)
        else:
            x = np.linspace(*xlim, n)

        data2plot = self(x)
        if kwargs['dx'] is None:
            kwargs['dx'] = np.ptp(x) / 14
        if kwargs['ylim'] is None:
            kwargs['ylim'] = (np.min(data2plot.d), np.max(data2plot.d))
        if kwargs['dy'] is None:
            kwargs['dy'] = np.ptp(data2plot.d) / 12

        self(x).plot(*pargs, **kwargs)

    def _polyprint(self, html=False):
        tr = {48: 8304,
              49: 185,
              50: 178,
              51: 179,
              52: 8308,
              53: 8309,
              54: 8310,
              55: 8311,
              56: 8312,
              57: 8313}
        # joiner[first, negative] = str
        joiner = {
            (True, True): '-',
            (True, False): '',
            (False, True): ' - ',
            (False, False): ' + '
        }

        result = []
        for power, coeff in reversed(list(enumerate(reversed(self.p.coeffs)))):
            j = joiner[not result, coeff < 0]
            coeff = abs(coeff)

            f = {0: '{0}{1}', 1: '{}{}·x'}.get(power, '{}{}·x{}')
            f0 = {0: '{}{}', 1: '{}x'}.get(power, '{0}x{2}')

            if coeff == 1:
                result.append(f0.format(j, coeff,
                                        str(power).translate(tr)
                                        if html is True
                                        else "^{}".format(power)))
            elif coeff != 0:
                result.append(f.format(j, coeff,
                                       str(power).translate(tr)
                                       if html is True
                                       else "^{}".format(power)))

        return ''.join(result) or '0'

    def __repr__(self):
        return self._polyprint()

    def _repr_html_(self):
        return self._polyprint(html=True)

    def resample(self, x):
        return mesh2d(x=mesh1d(x, label=self.x.label, unit=self.x.unit),
                      d=self.p(x), label=self.label, unit=self.unit)


class polymesh3d(object):
    """
    """

    def __init__(self, x_label=None, x_unit=None,
                 y_label=None, y_unit=None,
                 label=None, unit=None):

        self._x = axis(label=x_label, unit=x_unit)
        self._y = axis(label=y_label, unit=y_unit)
        self.z = axis(label=label, unit=unit)

        self._dtype = np.dtype([('x', 'f8'), ('p', object)])

    @property
    def x(self):
        return mesh1d(self.p.x,
                      label=self._x.label,
                      unit=self._x.unit)

    @property
    def y(self):
        """
        Describe the highest coefficent
        """
        return mesh1d(np.arange((max([len(_p.p.p.coeffs) for _p in self.p]))),
                      label=self._y.label,
                      unit=self._y.unit)

    def push(self, y, p):
        newElement = np.rec.array([(y, polymesh2d(p))], dtype=self._dtype)
        try:
            self.p = np.rec.array(np.append(newElement, self.p))
        except AttributeError:
            self.p = newElement

        # sort
        self.p = np.rec.array(self.p[self.p.x.argsort()])

        # This will be some pre-determined size
        shape = (len(self.x), len(self.y))
        self._d = np.zeros(shape, dtype=np.float64)

        for _x, _P in enumerate(self.p):
            self._d[_x, -len(_P.p.p) - 1:] = np.array(_P.p.p.coeffs)

        self._m3d = mesh3d(x=self.x, y=self.y, d=self._d,
                           label=self.z.label,
                           unit=self.z.label)

    def __call__(self, x=None, y=None):

        myPoly = polymesh2d(self._m3d(x=x).d,
                            x_label=self.y.label, x_unit=self.y.unit,
                            label=self.z.label, unit=self.z.unit)
        if y is None:
            return myPoly
        else:
            return myPoly(y)

    def plot(self, *pargs, **kwargs):
        import matplotlib.pyplot as plt
        for i, x in enumerate(self.p.x):
            self.p.p[i].plot(label="{} {}".format(str(x), self.x.unit),
                             *pargs, **kwargs)
        plt.legend(loc=2)

    def resample(self, y):
        res = mesh3d(x=mesh1d(**self.x.__dict__),
                     y=mesh1d(y, **self.y.__dict__),
                     label=self.z.label, unit=self.z.unit)
        for x, p in self.p:
            res.push(x, p(y).y)
        return res

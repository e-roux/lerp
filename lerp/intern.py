# -*- coding: utf-8 -*-
"""

"""

# Logging
import logging
import numpy as np
# set up logging to file - see previous section for more details

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
# set a format which is simpler for console use
formatter = logging.Formatter('%(levelname)s:%(name)s:%(lineno)d:%(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(console)


def deprecated(func):
    """Print a deprecation warning once on first use of the function.

    >>> @deprecated
    ... def f():
    ...     pass
    >>> f()
    f is deprecated
    """
    count = 0
    def wrapper(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            logger.warning(f"{func.__name__} is deprecated")
        return func(*args, **kwargs)
    return wrapper


def _myGrid(dx=None, dy=None):
    """
    plt.graph_paper = lambda dX = None, dY = None : myGrid(dX, dY)
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    fig = plt.gcf()
    for _i, _ax in enumerate(fig.axes):
        if _i == 0:
            if dx is not None:
                majorLocator = MultipleLocator(dx)
                _ax.axes.xaxis.set_major_locator(majorLocator)
            if dy is not None:
                majorLocator = MultipleLocator(dy)
                _ax.axes.yaxis.set_major_locator(majorLocator)

            _ax.axes.xaxis.set_minor_locator(AutoMinorLocator(n=10))
            _ax.axes.yaxis.set_minor_locator(AutoMinorLocator(n=10))
            lw = plt.rcParams['grid.linewidth']
            _ax.grid(b=True, which='major', linewidth=lw, alpha=1)
            _ax.grid(b=True, which='minor', linewidth=lw/2, alpha=0.6)

        else:
            _ax.grid(0)

def myPlot(func):
    """
    Permet de tracer un diagramme
    """

    def wrapper(data, *args, **kwargs):
        import matplotlib.pyplot as plt
        from lerp import mesh1d, mesh2d
        # --------------------------------------------------------------------
        # Options générales
        # --------------------------------------------------------------------
        rc = kwargs.pop('rc') if 'rc' in kwargs else False

        if rc is False:
            plt.style.use('ggplot' if 'ggplot' in plt.style.available
                          else 'default')

        if rc is True and 'graphpaper' in plt.style.available:
            plt.style.use('graphpaper')

        xlim = kwargs.pop('xlim') if 'xlim' in kwargs else None
        ylim = kwargs.pop('ylim') if 'ylim' in kwargs else None
        fileName = kwargs.pop('fileName') if 'fileName' in kwargs else None
        bokeh = kwargs.pop('bokeh') if 'bokeh' in kwargs else False
        newfigure = kwargs.pop('newfigure') if 'newfigure' in kwargs else False
        dX = kwargs.pop('dx') if 'dx' in kwargs else None
        dY = kwargs.pop('dy') if 'dy' in kwargs else None
        yaxis = kwargs.pop('yaxis') if 'yaxis' in kwargs else 'y1'

        if isinstance(data, mesh1d):
            Y = data
            X = mesh1d(np.arange(1, Y.size+1), "Support point", "-")
        elif isinstance(data, mesh2d):
            X = data.x
            Y = data.d
        else:
            raise Exception("No data to plot")

        if len(plt.get_fignums()) > 0 and not newfigure:
            fig = plt.gcf()
            _axe_tmp = plt.gca()

            if 'yaxis' in kwargs:
                yaxis = kwargs.pop('yaxis')
            if yaxis == 'y2':
                ax = _axe_tmp.twinx()
            else:
                ax = _axe_tmp
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        func(data, *args, **kwargs)

        # --------------------------------------------------------------------
        # Légende des axes
        # --------------------------------------------------------------------
        if X.label is not None:
            plt.xlabel(f"{X.label} [{X.unit}]")

        if data.label is not None:
            plt.ylabel(f"{data.label} [{data.unit}]")

        # --------------------------------------------------------------------
        # Tracé du graphe
        # --------------------------------------------------------------------
        myPlt = plt.plot(X, Y, *args, **kwargs)

        dX = np.ceil((float(X[-1])-float(X[0])) / 12)\
        if dX is None else float(dX)

        if dY is None:
            uY = np.float(ylim[1]-ylim[0] if ylim is not None
                          else Y.max() - Y.min())
            dY = np.ceil(float(np.abs(uY / 8)))
        else:
            dY = float(dY)

        # TODO : corriger
        # _myGrid(dX, dY)

        # Ces lignes car le matplotilibrc ne permet pas ces options
        [i.set_color("black") for i in plt.gca().get_xticklabels()]
        [i.set_color("black") for i in plt.gca().get_yticklabels()]

        alx = (fig.bbox.bounds[2]-fig.bbox.bounds[0]) / 12
        aly = (fig.bbox.bounds[3]-fig.bbox.bounds[1]) / 9

        plt.subplots_adjust(right=aly/alx)
        plt.xlim(xlim)

        _ylim = plt.ylim()

        if ylim is not None:
            plt.ylim(ylim)
#         else:
#             _y0 = np.sign(_ylim[0]) * dY * \
#             np.ceil(np.abs(np.float(_ylim[0]))/dY) \
#             if np.float(_ylim[0]) % dY != 0 else _ylim[0]
#
#             _y1 = np.sign(_ylim[1]) * dY * \
#             np.ceil(np.abs(np.float(_ylim[1]))/dY) \
#             if np.float(_ylim[1]) % dY != 0 else _ylim[1]
#
#             print(_y0, _y1)
#             plt.ylim(_y0, _y1)

        if bokeh is True:
            from bokeh import mpl
            from bokeh.plotting import output_notebook, show
            output_notebook()
            show(mpl.to_bokeh())

        if fileName is not None:
            print("Save file as " + fileName)
            plt.savefig(fileName, bbox_inches='tight')

        return myPlt
    return wrapper

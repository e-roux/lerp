.. api

.. module:: lerp

API
===


Apply to all mesh objects:

.. autosummary::

    mesh.max
    mesh.mean
    mesh.median
    mesh.min
    mesh.shape
    mesh.read_pickle
    mesh.to_pickle


mesh2d
------

.. autosummary::

   mesh2d.apply
   mesh2d.diff
   mesh2d.dropnan
   mesh2d.extrapolate
   mesh2d.gradient
   mesh2d.interpolate
   mesh2d.plot
   mesh2d.polyfit
   mesh2d.push
   mesh2d.read_clipboard
   mesh2d.resample
   mesh2d.step
   mesh2d.steps
   mesh2d.to_clipboard
   mesh2d.to_csv


mesh3d
------

.. autosummary::

   mesh3d.apply
   mesh3d.diff
   mesh3d.extrapolate
   mesh3d.from_pandas
   mesh3d.interpolate
   mesh3d.plot
   mesh3d.pop
   mesh3d.push
   mesh3d.read_clipboard
   mesh3d.reshape
   mesh3d.sort


mesh4d
------

.. autosummary::

   mesh4d.interpolate
   mesh4d.push
   mesh4d.reshape
   mesh4d.sort


polymesh2d
----------

.. autosummary::

   polymesh2d.resample


polymesh3d
----------

.. autosummary::

   polymesh3d.resample
   polymesh3d.plot
   polymesh3d.push

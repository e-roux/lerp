#ifndef MESH_H
#define MESH_H
// #include <Python.h>
// #define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
// #include <numpy/arrayobject.h>

#include "NDTable.h"


PyObject *
my_interp(PyObject *, PyObject *, PyObject *);

Mesh_h Mesh_FromXarray(PyObject *);


#endif
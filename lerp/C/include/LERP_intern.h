#ifndef INTERN_H
#define INTERN_H


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#include <numpy/arrayobject.h>


npy_intp
binary_search_with_guess(const npy_double key, const npy_double *arr,
                         npy_intp len, npy_intp guess);


NPY_NO_EXPORT PyObject *
my_interp(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict);

#endif
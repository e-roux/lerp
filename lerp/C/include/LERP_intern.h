#ifndef INTERN_H
#define INTERN_H
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>

// #include <Python.h>
// #define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
// #include <numpy/arrayobject.h>


npy_intp
binary_search_with_guess(const npy_double key, const npy_double *arr,
                         npy_intp len, npy_intp guess);


PyObject *
my_interp(PyObject *, PyObject *, PyObject *);


PyArrayObject* get_it(PyObject *);


#endif
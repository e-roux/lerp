#ifndef INTERN_H
#define INTERN_H
#include <numpy/arrayobject.h>
// #include <numpy/npy_math.h>

// #include <Python.h>
// #define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
// #include <numpy/arrayobject.h>



/**************************************************

Parameters
---------
key :   npy_double
		Element at which insertion index has to be found
arr :   Array of npy_double
        Breakpoints for which insertion index is beeing seeked
len :   npy_intp
        Array length
guess : str
        Extrapolation method

**************************************************/
npy_intp
binary_search_with_guess(const npy_double key, const npy_double *arr,
                         npy_intp len, npy_intp guess);


PyObject *
my_interp(PyObject *, PyObject *, PyObject *);


PyArrayObject* get_it(PyObject *);


#endif
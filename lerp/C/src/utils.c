/*
Comment

*/

#define PY_ARRAY_UNIQUE_SYMBOL UTILS_ARRAY_API
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#include <numpy/arrayobject.h>
#include "NumPyWrapper.h"

#define ARRAYD64(a) (PyArrayObject*) PyArray_ContiguousFromAny(a, NPY_DOUBLE, 0, 0)


// static PyObject *testmethod(PyObject *, PyObject *, PyObject *);


static PyObject *testmethod(PyObject *NPY_UNUSED(self),
                               PyObject *args,
                               PyObject *kwdict)
{
    /**************************************************

    Parameters
    ---------
    array :   Array object

    **************************************************/

    PyObject *output = NULL;       // returned value, Py_BuildValue of result_array
    PyObject *array = NULL;       // returned value, Py_BuildValue of result_array

    /**************************************************
    * Parse python call arguments
    **************************************************/
    static char *kwlist[] = {"array", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O", kwlist, &array)){
        return NULL;       
    }
    printf("Up to now ok 2\n");
    // printf("Refcount key: %zi\n", array->ob_refcnt);


    PyArrayObject *nparr = get_it(array);

    output = Py_BuildValue("O", (PyObject *) nparr);

    return output;
}


static PyMethodDef utils_methods[] = {
    {"testmethod", (PyCFunction) testmethod, METH_VARARGS | METH_KEYWORDS,
         "Un test"},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef utilsmodule = {
    PyModuleDef_HEAD_INIT,
    "utils",
    NULL,
    -1,
    utils_methods
};  

PyMODINIT_FUNC PyInit_utils(void)
{
    PyObject *mod = NULL;
    import_array();
    mod = PyModule_Create(&utilsmodule);
    return mod;
}




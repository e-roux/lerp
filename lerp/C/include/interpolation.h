/*

*/

#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#include <NDTable.h>

#ifdef __cplusplus
extern "C" {
#endif

// #define ARRAYD64(a) (PyArrayObject*) PyArray_FromAny(a, PyArray_DescrFromType(NPY_FLOAT64), 0, 0, 0, NULL)
#define ARRAYD64(a) (PyArrayObject*) PyArray_ContiguousFromAny(a, NPY_DOUBLE, 0, 0)


NDTable_h Mesh2NDTable(PyObject *mesh);


npy_intp evaluate_interpolation(NDTable_h mesh, const npy_double **params, npy_intp params_size,
                           NDTable_InterpMethod_t interp_method,
                           NDTable_ExtrapMethod_t extrap_method,
                           npy_double *result);

static PyObject *interpolation(PyObject *self, PyObject *args, PyObject *kwargs);


#ifdef __cplusplus
}
#endif

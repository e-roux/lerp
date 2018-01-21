
/*
PyObject* PyArray_SearchSorted(PyArrayObject* self, PyObject* values, NPY_SEARCHSIDE side, PyObject* perm)

https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.array.html#c.PyArray_SearchSorted

https://gist.github.com/saghul/1121260
*/

#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <algorithm.h>
#include <NDTable.h>


NDTable_h Mesh2NDTable(PyArrayObject *array, PyObject *breakpoints);

int evaluate_interpolation(NDTable_h mesh, const double **params, int ndimsparams,
                           NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method,
                           int resultSize, double *result);



NDTable_h Mesh2NDTable(PyArrayObject *array, PyObject *breakpoints){
    /* Fill the NDTable_t */

    NDTable_h output = (NDTable_h) malloc(sizeof(NDTable_t));

    npy_intp *shape_x = PyArray_DIMS(array),
             *strides_x = PyArray_STRIDES(array);

    output->ndim = PyArray_NDIM(array);
    for (int j=0; j < output->ndim; j++) {
        output->shape[j] = shape_x[j];
        output->strides[j] = strides_x[j];
        PyArrayObject *axis = (PyArrayObject *) PyList_GetItem(breakpoints, j);
        output->breakpoints[j] = PyArray_DATA(axis);
    }
    output->data = PyArray_DATA(array);
    output->size = PyArray_SIZE(array);
    output->itemsize = PyArray_ITEMSIZE(array);

    return output;

}

int evaluate_interpolation(
            NDTable_h mesh,
            const double **params,
            int ndimsparams,
            NDTable_InterpMethod_t interp_method,
            NDTable_ExtrapMethod_t extrap_method,
            int resultSize,
            double *result) {
    int i, j;
    double params_[32];

    for(i = 0; i < resultSize; i++) {
        for(j = 0; j < ndimsparams; j++) {
            params_[j] = params[j][i];
        }
        if(NDT_eval(mesh, ndimsparams, params_,
                    interp_method, extrap_method,
                    &result[i]) != NDTABLE_INTERPSTATUS_OK) {
            return -1;
        }
    }

    return 0;
}


static PyObject *
interpolation(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int err;
    PyObject *ret = NULL;

    //PyArrayObject *data_array = NULL;
//    double *breakpoints[MAX_NDIMS];

    PyArrayObject *result_array = NULL;
    npy_intp *shape_x;

    PyArrayObject *data_array = NULL;
    PyArrayObject *output = NULL;
    PyObject *bkpts = NULL;
    PyObject *targets = NULL;

    npy_intp resultSize;
    const char *interp_method = NULL,
               *extrap_method = NULL;

    static char *kwlist[] = {"data_array", "bkpts", "targets", "output", "interp", "extrap", NULL};

	/*
	Parse the parameters of a function that takes both positional 
    and keyword parameters into local variables. The keywords argument
    is a NULL-terminated array of keyword parameter names. Empty names
    denote positional-only parameters. Returns true on success; on failure,
    it returns false and raises the appropriate exception.
	*/
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|ss", kwlist, 
                                     &data_array, &bkpts, &targets, &output,
                                     &interp_method, &extrap_method)) {
		goto out;
    }

	if (interp_method == NULL) {
        interp_method = "interp";
    }

	if (extrap_method == NULL) {
        extrap_method = "hold";
    }

	// Check data array
    /*data_array = (PyArrayObject *) PyArray_FROM_OTF(data, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (data_array == NULL) {
        goto out;
    }*/

    if (!PyArray_Check(data_array)) {
        goto out;
    }

    NDTable_InterpMethod_t interpmethod = NDTABLE_INTERP_LINEAR;
    NDTable_ExtrapMethod_t extrapmethod = NDTABLE_EXTRAP_HOLD;

//    interp_m


    shape_x = PyArray_DIMS(data_array);
    
    // Check that data array has the same dim number as breakpoints
    Py_ssize_t bkptsdim = PySequence_Size(bkpts);

    if (bkptsdim != PyArray_NDIM(data_array)) {
        PyErr_SetString(PyExc_ValueError, "data and bkpts have different shapes");
        goto out;
    }

    // Create NDTable_h
    NDTable_h example = Mesh2NDTable(data_array, bkpts);


    //double **params;

    Py_ssize_t ndimsparams = PySequence_Size(targets);

    double *params[MAX_NDIMS];

    for (int j=0; j < ndimsparams; j++) {
        PyArrayObject *axis = (PyArrayObject *) PyList_GetItem(targets, j);
        params[j] = PyArray_DATA(axis);
        if (j==0) {
            resultSize = PyArray_SIZE(axis);
        }
    }
    
    printf("La valeur est %f et %lu, interp %d\n", params[0][2],
           ndimsparams, (int) resultSize);
    
    result_array = (PyArrayObject *) PyArray_SimpleNew(ndimsparams, &resultSize, NPY_FLOAT64);

    if (result_array == NULL) {
        goto out;
    }

    Py_XINCREF(result_array);

    
    err = evaluate_interpolation(example, (const double **) &params, example->ndim,
                                 interpmethod, extrapmethod,
                                 resultSize, (double *) PyArray_DATA(result_array));
    
    if (err != 0) {
        PyErr_Format(PyExc_ValueError, "Error %d occured in fancy_algorithm", err);
        goto out;
    }


	ret = Py_BuildValue("O", (PyObject *) result_array);
out:
//    Py_XDECREF(bkpts_array);
    Py_XDECREF(result_array);
    return ret;
}

static PyMethodDef interp_methods[] = {
    {"interpolation", (PyCFunction) interpolation, METH_VARARGS | METH_KEYWORDS,
         "Runs an algorithm defined in a local C file."},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef interpmodule = {
    PyModuleDef_HEAD_INIT,
    "interp",
    NULL,
    -1,
    interp_methods
};

PyMODINIT_FUNC
PyInit_interp(void)
{
    PyObject *mod = NULL;
    import_array();
    mod = PyModule_Create(&interpmodule);
    return mod;
}



    /*
    PyObject_Print(self, stdout, 0);
    fprintf(stdout, "\n");
    PyObject_Print(args, stdout, 0);
    fprintf(stdout, "\n");
    PyObject_Print(kwargs, stdout, 0);
    fprintf(stdout, "\n");
    */

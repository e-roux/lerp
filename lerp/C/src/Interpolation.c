
/*
PyObject* PyArray_SearchSorted(PyArrayObject* self, PyObject* values, NPY_SEARCHSIDE side, PyObject* perm)

https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.array.html#c.PyArray_SearchSorted

https://gist.github.com/saghul/1121260


Reference count: http://edcjones.tripod.com/refcount.html
*/

#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <NDTable.h>


#define ARRAYD64(a) (PyArrayObject*) PyArray_FromAny(a, PyArray_DescrFromType(NPY_FLOAT64), 0, 0, 0, NULL)


NDTable_h Mesh2NDTable(PyObject *mesh);


int evaluate_interpolation(NDTable_h mesh, const double **params, int params_size,
                           NDTable_InterpMethod_t interp_method,
                           NDTable_ExtrapMethod_t extrap_method,
                           double *result);

static PyObject *interpolation(PyObject *self, PyObject *args, PyObject *kwargs);



NDTable_h Mesh2NDTable(PyObject *mesh){

    // TODO : check that mesh is subclass of xarray
    // PyObject_IsSubclass(mesh , (PyObject*))
    /*    if (!PyArray_Check(data_array))
            goto out;
    */
    // Cast data to double
    PyArrayObject *array = ARRAYD64(PyObject_GetAttrString(mesh, "data"));
   
    // coords
    PyObject *coords = PyObject_GetAttrString(mesh, "coords");
    PyObject *variable = PyObject_GetAttrString(mesh, "variable");
    PyObject *coords_list = PyObject_GetAttrString(mesh, "dims");

    PyObject *key;

    // Check that data array has the same dim number as coords
    if (PySequence_Length(coords_list) != PyArray_NDIM(array)) {
        PyErr_SetString(PyExc_ValueError, "Data and bkpts have different shapes");
        // todo : exit
    }

    NDTable_h output = (NDTable_h) malloc(sizeof(NDTable_t));

    npy_intp *shape_x = PyArray_DIMS(array);

    output->ndim = PyArray_NDIM(array);
    for (Py_ssize_t j=0; j < output->ndim; j++) {
        output->shape[j] = shape_x[j];

        if (PyTuple_Check(coords_list)) {
            /* PySequence_GetItem INCREFs key. */
            key = PyTuple_GetItem(coords_list, j);
        }

       PyObject *axis = PyObject_GetAttrString(mesh, (char *)PyUnicode_AS_DATA(key));

       Py_DECREF(key);

        PyArrayObject *coords_tmp =  ARRAYD64(axis);
        output->coords[j] = PyArray_DATA(coords_tmp);

        Py_DECREF(axis);
    }
    output->data = PyArray_DATA(array);
    output->size = PyArray_SIZE(array);
    output->itemsize = PyArray_ITEMSIZE(array);
//    output->interpmethod = *interp_linear;

    Py_DECREF(coords_list);
    Py_DECREF(coords);
    Py_DECREF(array);

    return output;

}


NDTable_InterpMethod_t get_interp_method(char *method) {

    // Set interpolation method, default to linear
    NDTable_InterpMethod_t interpmethod;

    if (strcmp(method, "hold") == 0) {
        interpmethod = NDTABLE_INTERP_HOLD;
    } 
    else if (strcmp(method, "nearest") == 0) {
        interpmethod = NDTABLE_INTERP_NEAREST;
    } 
    else if (strcmp(method, "linear") == 0) {
        interpmethod = NDTABLE_INTERP_LINEAR;
    } 
    else if (strcmp(method, "akima") == 0) {
        interpmethod = NDTABLE_INTERP_AKIMA;
    } 
    else if (strcmp(method, "fritsch_butland") == 0) {
        interpmethod = NDTABLE_INTERP_FRITSCH_BUTLAND;
    }     
    else if (strcmp(method, "steffen") == 0) {
        interpmethod = NDTABLE_INTERP_STEFFEN;
    }
    else {
        interpmethod = NDTABLE_INTERP_LINEAR;
    }

    return interpmethod;
}

NDTable_ExtrapMethod_t get_extrap_method(char *method) {

    // Set extrapolation method, default to hold
    NDTable_ExtrapMethod_t extrapmethod;

    if (strcmp(method, "hold") == 0) {
        extrapmethod = NDTABLE_EXTRAP_HOLD;
    } 
    else if (strcmp(method, "linear") == 0) {
        extrapmethod = NDTABLE_EXTRAP_LINEAR;
    } 
    else {
        extrapmethod = NDTABLE_EXTRAP_HOLD;
    }

    return extrapmethod;
}



int evaluate_interpolation( NDTable_h mesh, const double **params, int params_size,
            NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method,
            double *result)
{

    int      index[NPY_MAXDIMS]; // the subscripts
    int      nsubs[NPY_MAXDIMS]; // the neighboring subscripts
    double   derivatives[NPY_MAXDIMS];
    double   weigths[NPY_MAXDIMS]; // the weights for the interpolation

    // if the dataset is scalar return the value
    if (mesh->ndim == 0) {
        *result = mesh->data[0];
        return NDTABLE_INTERPSTATUS_OK;
    }

    // TODO: add null check

    // Iteration over each points
    for(int i = 0; i < params_size; i++) {
        // set array of parameter in each direction
        for(int j = 0; j < mesh->ndim; j++) {
            // Find index and weights
            NDTable_find_index(params[j][i], mesh->shape[j], mesh->coords[j],
                               &index[j], &weigths[j]);
        }
        int status = NDT_eval_internal(mesh, weigths, index, nsubs, 0, interp_method,
                                        extrap_method, &result[i], derivatives);

        if(status != NDTABLE_INTERPSTATUS_OK) {
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

    PyArrayObject *result_array = NULL;

    PyObject *mesh = NULL;
    PyObject *targets = NULL;

    npy_intp params_size;

    // Set interpolation option, default to linear
    // Set extrapolation option, default to hold
    char *interp_method = "linear",
         *extrap_method = "hold";

    static char *kwlist[] = {"mesh", "targets", "interp", "extrap", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ss", kwlist, &mesh, &targets,
                                     &interp_method, &extrap_method))
		goto out;

    NDTable_InterpMethod_t interpmethod = get_interp_method(interp_method);
    NDTable_ExtrapMethod_t extrapmethod = get_extrap_method(extrap_method);

    // Create NDTable_h
    NDTable_h example = Mesh2NDTable(mesh);

    Py_ssize_t ndimsparams = PySequence_Size(targets);

    double *params[NPY_MAXDIMS];

    for (int j=0; j < example->ndim; j++) {
        PyArrayObject *axis = ARRAYD64(PyList_GetItem(targets, j));
        params[j] = PyArray_DATA(axis);

        if (j==0) {
            params_size = PyArray_SIZE(axis);
            result_array = (PyArrayObject *) PyArray_NewLikeArray(axis, NPY_CORDER, NULL, 1);
        }
    }

    if (result_array == NULL) {
        goto out;
    }

    err = evaluate_interpolation(example, (const double **) &params, params_size,
                                 interpmethod, extrapmethod,
                                 (double *) PyArray_DATA(result_array));

    if (err != 0) {
        PyErr_Format(PyExc_ValueError, "Error %d occured in fancy_algorithm", err);
        goto out;
    }
    Py_XINCREF(result_array);


	ret = Py_BuildValue("O", (PyObject *) result_array);
 out:
    Py_XDECREF(result_array);
    return ret;
}

int evaluate_derivate(
            NDTable_h mesh,
            const double **params,
            int params_size,
            NDTable_InterpMethod_t interp_method,
            NDTable_ExtrapMethod_t extrap_method,
            double *result) 
{
    int      index[NPY_MAXDIMS]; // the subscripts
    int      nsubs[NPY_MAXDIMS]; // the neighboring subscripts
    double   derivatives[NPY_MAXDIMS];
    double   weigths[NPY_MAXDIMS]; // the weights for the interpolation

    // if the dataset is scalar return the value
    if (mesh->ndim == 0) {
        *result = mesh->data[0];
        return NDTABLE_INTERPSTATUS_OK;
    }

    // Iteration over each points
    for(int i = 0; i < params_size; i++) {
        // set array of parameter in each direction
        for(int j = 0; j < mesh->ndim; j++) {
            // Find index and weights
            NDTable_find_index(params[j][i], mesh->shape[j], mesh->coords[j],
                               &index[j], &weigths[j]);
        }
        int status = NDT_eval_internal(mesh, weigths, index, nsubs, 0, interp_method,
                                        extrap_method, &result[i], derivatives);

        if(status != NDTABLE_INTERPSTATUS_OK) {
            return -1;
        }

    /*        *result[i] = 0.0;

            for (int k = 0; k < mesh->ndim; k++) {
                *result[i] += params[j][i] * derivatives[i];
            }
    */

    }

    return 0;

}

static PyObject *
derivate(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int err;
    PyObject *ret = NULL;

    PyArrayObject *result_array = NULL;

    // PyArrayObject *data_array = NULL;
    PyObject *mesh = NULL;
    PyObject *bkpts = NULL;
    PyObject *targets = NULL;

    int bkptsMaxLength = 1;

    npy_intp params_size;

    // Set interpolation option, default to linear
    // Set extrapolation option, default to hold
    char *interp_method = "linear",
         *extrap_method = "hold";

    static char *kwlist[] = {"mesh", "bkpts", "targets", "interp", "extrap", NULL};

    /*
    Parse the parameters of a function that takes both positional
    and keyword parameters into local variables. The keywords argument
    is a NULL-terminated array of keyword parameter names. Empty names
    denote positional-only parameters. Returns true on success; on failure,
    it returns false and raises the appropriate exception.
    */
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|ss", kwlist,
                                     &mesh, &bkpts, &targets,
                                     &interp_method, &extrap_method)) {
        goto out;
    }

    // Check data array
    /*data_array = (PyArrayObject *) PyArray_FROM_OTF(data, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (data_array == NULL) {
        goto out;
    }*/


//    bkpts = (PyArrayObject*)PyObject_GetAttrString(mesh, "data");


/*    if (!PyArray_Check(data_array)) {
        goto out;
    }*/
   
    NDTable_InterpMethod_t interpmethod = get_interp_method(interp_method);
    NDTable_ExtrapMethod_t extrapmethod = get_extrap_method(extrap_method);

    // Check that data array has the same dim number as coords
/*    Py_ssize_t bkptsdim = PySequence_Size(bkpts);

    if (bkptsdim != PyArray_NDIM(data_array)) {
        PyErr_SetString(PyExc_ValueError, "Data and bkpts have different shapes");
        goto out;
    }
*/
    // Check that the bkpts can be broadcasted in an array
    // PyObject_HasAttr


    // Create NDTable_h
    NDTable_h example = Mesh2NDTable(mesh);

    //double **params;

    Py_ssize_t ndimsparams = PySequence_Size(targets);

    double *params[NPY_MAXDIMS];

    //    PyArrayObject *result_array;

    for (int j=0; j < example->ndim; j++) {
        PyArrayObject *axis = (PyArrayObject *) PyList_GetItem(targets, j);
        params[j] = PyArray_DATA(axis);
        if (j==0) {
            params_size = PyArray_SIZE(axis);
            result_array = (PyArrayObject *) PyArray_NewLikeArray(axis, NPY_CORDER, NULL, 1);
        }
    }

    /*    printf("La valeur est %f et %lu, interp %d\n", params[0][2],
           ndimsparams, (int) params_size);*/

    /*result_array = (PyArrayObject *) PyArray_SearchSorted(PyList_GetItem(bkpts, 0),
                                                          PyList_GetItem(targets, 0),
                                                          NPY_SEARCHLEFT, NULL);
    */

    if (result_array == NULL) {
        goto out;
    }

    err = evaluate_interpolation(example, (const double **) &params, params_size,
                                 interpmethod, extrapmethod,
                                 (double *) PyArray_DATA(result_array));

    if (err != 0) {
        PyErr_Format(PyExc_ValueError, "Error %d occured in fancy_algorithm", err);
        goto out;
    }
    Py_XINCREF(result_array);


    ret = Py_BuildValue("O", (PyObject *) result_array);
 out:
    Py_XDECREF(result_array);
    return ret;
}

static PyMethodDef interpolation_methods[] = {
    {"interpolation", (PyCFunction) interpolation, METH_VARARGS | METH_KEYWORDS,
         "Interpolation."},
    {"derivate", (PyCFunction) derivate, METH_VARARGS | METH_KEYWORDS,
         "Runs an algorithm defined in a local C file."},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef interpolationmodule = {
    PyModuleDef_HEAD_INIT,
    "interpolation",
    NULL,
    -1,
    interpolation_methods
};

PyMODINIT_FUNC
PyInit_interpolation(void)
{
    PyObject *mod = NULL;
    import_array();
    mod = PyModule_Create(&interpolationmodule);
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

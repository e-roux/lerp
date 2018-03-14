
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
#include <interpolation.h>

#define error_converting(x)  (((x) == -1) && PyErr_Occurred())



/** @brief find index of a sorted array such that arr[i] <= key < arr[i + 1].
 *
 * If an starting index guess is in-range, the array values around this
 * index are first checked.  This allows for repeated calls for well-ordered
 * keys (a very common case) to use the previous index as a very good guess.
 *
 * If the guess value is not useful, bisection of the array is used to
 * find the index.  If there is no such index, the return values are:
 *     key < arr[0] -- -1
 *     key == arr[len - 1] -- len - 1
 *     key > arr[len - 1] -- len
 * The array is assumed contiguous and sorted in ascending order.
 *
 * @param key key value.
 * @param arr contiguous sorted array to be searched.
 * @param len length of the array.
 * @param guess initial guess of index
 * @return index
 */
#define LIKELY_IN_CACHE_SIZE 8

static npy_intp
binary_search_with_guess(const npy_double key, const npy_double *arr,
                         npy_intp len, npy_intp guess)
{
    npy_intp imin = 0;
    npy_intp imax = len;

    /* Handle keys outside of the arr range first */
    if (key > arr[len - 1]) {
        return len;
    }
    else if (key < arr[0]) {
        return -1;
    }

    /*
     * If len <= 4 use linear search.
     * From above we know key >= arr[0] when we start.
     */
    if (len <= 4) {
        npy_intp i;

        for (i = 1; i < len && key >= arr[i]; ++i);
        return i - 1;
    }

    if (guess > len - 3) {
        guess = len - 3;
    }
    if (guess < 1)  {
        guess = 1;
    }

    /* check most likely values: guess - 1, guess, guess + 1 */
    if (key < arr[guess]) {
        if (key < arr[guess - 1]) {
            imax = guess - 1;
            /* last attempt to restrict search to items in cache */
            if (guess > LIKELY_IN_CACHE_SIZE &&
                        key >= arr[guess - LIKELY_IN_CACHE_SIZE]) {
                imin = guess - LIKELY_IN_CACHE_SIZE;
            }
        }
        else {
            /* key >= arr[guess - 1] */
            return guess - 1;
        }
    }
    else {
        /* key >= arr[guess] */
        if (key < arr[guess + 1]) {
            return guess;
        }
        else {
            /* key >= arr[guess + 1] */
            if (key < arr[guess + 2]) {
                return guess + 1;
            }
            else {
                /* key >= arr[guess + 2] */
                imin = guess + 2;
                /* last attempt to restrict search to items in cache */
                if (guess < len - LIKELY_IN_CACHE_SIZE - 1 &&
                            key < arr[guess + LIKELY_IN_CACHE_SIZE]) {
                    imax = guess + LIKELY_IN_CACHE_SIZE;
                }
            }
        }
    }

    /* finally, find index by bisection */
    while (imin < imax) {
        const npy_intp imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid]) {
            imin = imid + 1;
        }
        else {
            imax = imid;
        }
    }
    return imin - 1;
}

#undef LIKELY_IN_CACHE_SIZE


NPY_NO_EXPORT PyObject *
my_interp(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{

    PyObject *fp, *xp, *x;
    PyObject *left = NULL, *right = NULL;
    PyArrayObject *afp = NULL, *axp = NULL, *ax = NULL, *af = NULL;
    npy_intp i, lenx, lenxp;
    npy_double lval, rval;
    const npy_double *dy, *dx, *dz;
    npy_double *dres, *slopes = NULL;

    static char *kwlist[] = {"x", "xp", "fp", "left", "right", NULL};

    NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "OOO|OO:interp", kwlist,
                                     &x, &xp, &fp, &left, &right)) {
        return NULL;
    }

    afp = (PyArrayObject *)PyArray_ContiguousFromAny(fp, NPY_DOUBLE, 1, 1);
    if (afp == NULL) {
        return NULL;
    }
    axp = (PyArrayObject *)PyArray_ContiguousFromAny(xp, NPY_DOUBLE, 1, 1);
    if (axp == NULL) {
        goto fail;
    }
    ax = (PyArrayObject *)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 0, 0);
    if (ax == NULL) {
        goto fail;
    }
    lenxp = PyArray_SIZE(axp);
    if (lenxp == 0) {
        PyErr_SetString(PyExc_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    if (PyArray_SIZE(afp) != lenxp) {
        PyErr_SetString(PyExc_ValueError,
                "fp and xp are not of the same length.");
        goto fail;
    }

    af = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ax),
                                            PyArray_DIMS(ax), NPY_DOUBLE);
    if (af == NULL) {
        goto fail;
    }
    lenx = PyArray_SIZE(ax);

    dy = (const npy_double *)PyArray_DATA(afp);
    dx = (const npy_double *)PyArray_DATA(axp);
    dz = (const npy_double *)PyArray_DATA(ax);
    dres = (npy_double *)PyArray_DATA(af);
    /* Get left and right fill values. */
    if ((left == NULL) || (left == Py_None)) {
        lval = dy[0];
    }
    else {
        lval = PyFloat_AsDouble(left);
        if (error_converting(lval)) {
            goto fail;
        }
    }
    if ((right == NULL) || (right == Py_None)) {
        rval = dy[lenxp - 1];
    }
    else {
        rval = PyFloat_AsDouble(right);
        if (error_converting(rval)) {
            goto fail;
        }
    }

    /* binary_search_with_guess needs at least a 3 item long array */
    if (lenxp == 1) {
        const npy_double xp_val = dx[0];
        const npy_double fp_val = dy[0];

        NPY_BEGIN_THREADS_THRESHOLDED(lenx);
        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];
            dres[i] = (x_val < xp_val) ? lval :
                                         ((x_val > xp_val) ? rval : fp_val);
        }
        NPY_END_THREADS;
    }
    else {
        npy_intp j = 0;

        /* only pre-calculate slopes if there are relatively few of them. */
        if (lenxp <= lenx) {
            slopes = PyArray_malloc((lenxp - 1) * sizeof(npy_double));
            if (slopes == NULL) {
                goto fail;
            }
        }

        NPY_BEGIN_THREADS;

        if (slopes != NULL) {
            for (i = 0; i < lenxp - 1; ++i) {
                slopes[i] = (dy[i+1] - dy[i]) / (dx[i+1] - dx[i]);
            }
        }

        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];

            if (npy_isnan(x_val)) {
                dres[i] = x_val;
                continue;
            }

            j = binary_search_with_guess(x_val, dx, lenxp, j);  
            if (j == -1) {
                dres[i] = lval;
            }
            else if (j == lenxp) {
                dres[i] = rval;
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];
            }
            else {
                const npy_double slope = (slopes != NULL) ? slopes[j] :
                                         (dy[j+1] - dy[j]) / (dx[j+1] - dx[j]);
                dres[i] = slope*(x_val - dx[j]) + dy[j];
            }
        }
        NPY_END_THREADS;
    }

    PyArray_free(slopes);
    Py_DECREF(afp);
    Py_DECREF(axp);
    Py_DECREF(ax);
    return PyArray_Return(af);

fail:
    Py_XDECREF(afp);
    Py_XDECREF(axp);
    Py_XDECREF(ax);
    Py_XDECREF(af);
    return NULL;
}


NDTable_h Mesh2NDTable(PyObject *mesh){

    // TODO : check that mesh is subclass of xarray
    // PyObject_IsSubclass(mesh , (PyObject*))
    /*    if (!PyArray_Check(data_array))
            goto out;
    */
    // Cast data to npy_double
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

      // Py_DECREF(key);

        PyArrayObject *coords_tmp =  ARRAYD64(axis);
        output->coords[j] = PyArray_DATA(coords_tmp);

        Py_DECREF(axis);
    }
    output->data = PyArray_DATA(array);
    output->size = PyArray_SIZE(array);
    output->itemsize = PyArray_ITEMSIZE(array);
    // output->interpmethod = *interp_linear;


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


npy_intp evaluate_interpolation(NDTable_h mesh, const npy_double **params,
                                npy_intp params_size,
                                NDTable_InterpMethod_t interp_method,
                                NDTable_ExtrapMethod_t extrap_method,
                                npy_double *result)
{

    npy_intp      index[NPY_MAXDIMS]; // the subscripts
    npy_intp      nsubs[NPY_MAXDIMS]; // the neighboring subscripts
    npy_double   derivatives[NPY_MAXDIMS];
    npy_double   weigths[NPY_MAXDIMS]; // the weights for the interpolation

    npy_intp i, j, _cache;
    // if the dataset is scalar return the value
    if (mesh->ndim == 0) {
        *result = mesh->data[0];
        return NDTABLE_INTERPSTATUS_OK;
    }

    // TODO: add null check
   NPY_BEGIN_THREADS_DEF;

   NPY_BEGIN_THREADS_THRESHOLDED(params_size);

    // Iteration over each points
    for(i = 0; i < params_size; i++) {
        // set array of parameter in each direction
        for(j = 0; j < mesh->ndim; j++) {
            // Find index and weights
            // NDTable_find_index(params[j][i], mesh->shape[j], mesh->coords[j],
            //                    &index[j], &weigths[j]);

            // index[j] = binary_search_with_guess(params[j][i], mesh->coords[j],
            //                                     mesh->shape[j], index[j]);

            _cache = binary_search_with_guess(params[j][i], mesh->coords[j],
                                              mesh->shape[j], _cache);
            index[j] = _cache;

            weigths[j] = (params[j][_cache] - mesh->coords[j][_cache]) /
                         (mesh->coords[j][_cache+1] - mesh->coords[j][_cache]);
        }
        npy_intp status = NDT_eval_internal(mesh, weigths, index, nsubs, 0, interp_method,
                                        extrap_method, &result[i], derivatives);

        if(status != NDTABLE_INTERPSTATUS_OK) {
            return -1;
        }
    }
    
    NPY_END_THREADS;

    return 0;
}

static PyObject *
interpolation(PyObject *self, PyObject *args, PyObject *kwargs)
{
    npy_intp err;
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

    npy_double *params[NPY_MAXDIMS];

    for (npy_intp j=0; j < example->ndim; j++) {
        PyArrayObject *axis = ARRAYD64(PyList_GetItem(targets, j));
        params[j] = PyArray_DATA(axis);

        if (j==0) {
            params_size = PyArray_SIZE(axis);
            result_array = (PyArrayObject *) PyArray_NewLikeArray(axis, NPY_CORDER, NULL, 1);
        }
        //Py_DECREF(axis);
    }


    if (result_array == NULL) {
        goto out;
    }

    err = evaluate_interpolation(example, (const npy_double **) &params, params_size,
                                 interpmethod, extrapmethod,
                                 (npy_double *) PyArray_DATA(result_array));

    if (err != 0) {
        PyErr_Format(PyExc_ValueError, "Error %d occured in fancy_algorithm", err);
        goto out;
    }

	ret = Py_BuildValue("O", (PyObject *) result_array);
 out:
    // Py_DECREF(result_array);
    return ret;
}

npy_intp evaluate_derivate(
            NDTable_h mesh,
            const npy_double **params,
            npy_intp params_size,
            NDTable_InterpMethod_t interp_method,
            NDTable_ExtrapMethod_t extrap_method,
            npy_double *result) 
{
    npy_intp      index[NPY_MAXDIMS]; // the subscripts
    npy_intp      nsubs[NPY_MAXDIMS]; // the neighboring subscripts
    npy_double   derivatives[NPY_MAXDIMS];
    npy_double   weigths[NPY_MAXDIMS]; // the weights for the interpolation

    // if the dataset is scalar return the value
    if (mesh->ndim == 0) {
        *result = mesh->data[0];
        return NDTABLE_INTERPSTATUS_OK;
    }

    // Iteration over each points
    for(npy_intp i = 0; i < params_size; i++) {
        // set array of parameter in each direction
        for(npy_intp j = 0; j < mesh->ndim; j++) {
            // Find index and weights
            NDTable_find_index(params[j][i], mesh->shape[j], mesh->coords[j],
                               &index[j], &weigths[j]);
        }
        npy_intp status = NDT_eval_internal(mesh, weigths, index, nsubs, 0, interp_method,
                                        extrap_method, &result[i], derivatives);

        if(status != NDTABLE_INTERPSTATUS_OK) {
            return -1;
        }

    /*        *result[i] = 0.0;

            for (npy_intp k = 0; k < mesh->ndim; k++) {
                *result[i] += params[j][i] * derivatives[i];
            }
    */

    }

    return 0;
}

static PyObject *
derivate(PyObject *self, PyObject *args, PyObject *kwargs)
{
    npy_intp err;
    PyObject *ret = NULL;

    PyArrayObject *result_array = NULL;

    // PyArrayObject *data_array = NULL;
    PyObject *mesh = NULL;
    PyObject *bkpts = NULL;
    PyObject *targets = NULL;

    npy_intp bkptsMaxLength = 1;

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

    //npy_double **params;

    Py_ssize_t ndimsparams = PySequence_Size(targets);

    npy_double *params[NPY_MAXDIMS];

    //    PyArrayObject *result_array;

    for (npy_intp j=0; j < example->ndim; j++) {
        PyArrayObject *axis = (PyArrayObject *) PyList_GetItem(targets, j);
        params[j] = PyArray_DATA(axis);
        if (j==0) {
            params_size = PyArray_SIZE(axis);
            result_array = (PyArrayObject *) PyArray_NewLikeArray(axis, NPY_CORDER, NULL, 1);
        }
    }

    /*    printf("La valeur est %f et %lu, interp %d\n", params[0][2],
           ndimsparams, (npy_intp) params_size);*/

    /*result_array = (PyArrayObject *) PyArray_SearchSorted(PyList_GetItem(bkpts, 0),
                                                          PyList_GetItem(targets, 0),
                                                          NPY_SEARCHLEFT, NULL);
    */

    if (result_array == NULL) {
        goto out;
    }

    err = evaluate_interpolation(example, (const npy_double **) &params, params_size,
                                 interpmethod, extrapmethod,
                                 (npy_double *) PyArray_DATA(result_array));

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
    {"my_interp", (PyCFunction) my_interp, METH_VARARGS | METH_KEYWORDS,
         "my_interp."},         
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


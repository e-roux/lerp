
/*
PyObject* PyArray_SearchSorted(PyArrayObject* self, PyObject* values, NPY_SEARCHSIDE side, PyObject* perm)

https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.array.html#c.PyArray_SearchSorted

https://gist.github.com/saghul/1121260


Reference count: http://edcjones.tripod.com/refcount.html
*/
#include <interpolation.h>
#include <structmember.h>

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

npy_intp myfunction(npy_intp elt) {
    printf("%i\n", (int)elt);
    return elt;
}



NDTable_h Mesh2NDTable(PyObject *mesh){


                       // NDTable_InterpMethod_t *interpmethod,
                       // NDTable_ExtrapMethod_t *extrapmethod 

    // TODO : check that mesh is subclass of xarray
    // PyObject_IsSubclass(mesh , (PyObject*))
    /*    if (!PyArray_Check(data_array))
            goto out;
    */
    // Cast data to npy_double
    PyArrayObject *array = ARRAYD64(
        PyObject_GetAttrString(mesh, "data"));
   
    // coords
    PyObject *coords = PyObject_GetAttrString(mesh, "coords");
    PyObject *variable = PyObject_GetAttrString(mesh, "variable");
    PyObject *coords_list = PyObject_GetAttrString(mesh, "dims");

    PyObject *key;

    // Check that data array has the same dim number as coords
    if (PySequence_Length(coords_list) != PyArray_NDIM(array)) {
        PyErr_SetString(PyExc_ValueError, 
            "Data and bkpts have different shapes");
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

       PyObject *axis = PyObject_GetAttrString(mesh,
            (char *)PyUnicode_AS_DATA(key));

      // Py_DECREF(key);

        PyArrayObject *coords_tmp =  ARRAYD64(axis);
        output->coords[j] = PyArray_DATA(coords_tmp);

        Py_DECREF(axis);
    }
    output->data = PyArray_DATA(array);
    output->size = PyArray_SIZE(array);
    output->itemsize = PyArray_ITEMSIZE(array);
    output->interpmethod = &myfunction; // *interp_linear;
    // output->interpmethod = interpmethod; // *interp_linear;
    // output->extrapmethod = extrapmethod; // *interp_linear;


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



static PyObject *_raise_error(PyObject *module) {

    PyErr_SetString(PyExc_ValueError, "Ooops.");
    return NULL;
}


static PyObject *interpolation(PyObject *NPY_UNUSED(self),
                               PyObject *args,
                               PyObject *kwdict)
{
    /**************************************************

    Parameters
    ---------
    mesh :    Mesh object
              Labeled nd-array  
    targets : Sequence of array
              Elements for which interpolation values are computed
    inter :   str
              Interpolation method
    extrap :  str
              Extrapolation method

    **************************************************/

    PyObject *ret = NULL; // returned value, Py_BuildValue of result_array
    PyArrayObject *result_array = NULL;

    PyObject *mesh = NULL; // function paramters from Python code
    PyObject *targets = NULL; // function paramters from Python code

    npy_intp result_array_size;

    npy_intp      index[NPY_MAXDIMS]; // the subscripts
    npy_intp      nsubs[NPY_MAXDIMS]; // the neighboring subscripts
    npy_double    derivatives[NPY_MAXDIMS];
    npy_double    weigths[NPY_MAXDIMS]; // the weights for the interpolation

    NDTable_h table;

    npy_double *result_data;

    npy_intp i, j, _cache;

    const npy_double *dx;

    /**************************************************
    Set interpolation default to linear
    Set extrapolation default to hold
    **************************************************/
    char *interp_method = "linear",
         *extrap_method = "hold";

    /**************************************************
    * Parse python call arguments
    **************************************************/
    static char *kwlist[] = {"mesh", "targets", "interp",
                             "extrap", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "OO|ss", kwlist,
                                     &mesh, &targets,
                                     &interp_method, &extrap_method)){
        return NULL;       
    }

    /**************************************************
    * Check interpolation and extrapolation method
    **************************************************/
    NDTable_InterpMethod_t interpmethod = get_interp_method(interp_method);
    NDTable_ExtrapMethod_t extrapmethod = get_extrap_method(extrap_method);

    /**************************************************
    * Create NDTable_h
    **************************************************/    
    table = Mesh2NDTable(mesh); // , *interpmethod, *extrapmethod);

    /**************************************************
    * Build targets and shape plausibility check
        - first build : only accept if all target
          shapes are identical
    **************************************************/    
    NDTargets_h mytargets = (NDTargets_h) malloc(sizeof(NDTargets_t));
    mytargets->ndim = PySequence_Size(targets);

    if (mytargets->ndim == 0) {
        goto fail;
    }

    // 

    for (Py_ssize_t j=0; j < mytargets->ndim; j++) {
        mytargets->coords[j] = ARRAYD64(PyList_GetItem(targets, j));

        printf("%zd\n", PyArray_SIZE(mytargets->coords[j]));
        
        if (j==0) {
            result_array = (PyArrayObject *) PyArray_NewLikeArray(mytargets->coords[0],
                NPY_CORDER, NULL, 1);
        }
        else {
            if(PyArray_SIZE(mytargets->coords[j]) != 
               PyArray_SIZE(mytargets->coords[0])) {
                PyErr_Format(PyExc_ValueError,
                    "All target breaking points must be the same size.");
                goto out;
            }
        }
    }

    // mesh and targets must have the same shape.
    if(mytargets->ndim != table->ndim) {
        PyErr_Format(PyExc_ValueError,
            "Targets shape and mesh coords have different shapes.");
        goto out;
    }

    result_data = PyArray_DATA(result_array);
    result_array_size = PyArray_SIZE(result_array);
    /**************************************************
    * Create NDTable_h
    **************************************************/
    // if the dataset is scalar return the value
    if (table->ndim == 0) {
        result_data = &table->data[0];
    }
    else {
        // TODO: add null check
       NPY_BEGIN_THREADS_DEF;
       NPY_BEGIN_THREADS_THRESHOLDED(result_array_size);
  
        // START_TIMING;
        // Iteration over each points
        for(i = 0; i < result_array_size; i++) {

            // search index for interpolation and calculate weight
            for(j = 0; j < table->ndim; j++) {
                dx = (const npy_double *)PyArray_DATA(mytargets->coords[j]);

                _cache = binary_search_with_guess(dx[i], 
                                                  table->coords[j],
                                                  table->shape[j],
                                                  _cache);
                index[j] = _cache;

                weigths[j] = (dx[_cache] - table->coords[j][_cache]) /
                             (table->coords[j][_cache+1] - table->coords[j][_cache]);
            }
            npy_intp status = NDT_eval_internal(
                table, weigths, index, nsubs, 0,
                interpmethod, extrapmethod, &result_data[i]);
 
            if(status != NDTABLE_INTERPSTATUS_OK) {
                PyErr_Format(PyExc_ValueError,
                    "Error %d occured in fancy_algorithm", status);
                goto out;
            }
        }

        // END_TIMING;
        
        NPY_END_THREADS;

    }

    /**************************************************
    * Check interpolation and extrapolation method
    **************************************************/
	ret = Py_BuildValue("O", (PyObject *) result_array);

    for (npy_intp j=0; j < mytargets->ndim; j++) {
       Py_XDECREF(mytargets->coords[j]);
    }

    out:
       // Py_DECREF(result_array);
        return ret;
    fail:
        return NULL;        
}


static PyMethodDef interpolation_methods[] = {
    {"interpolation", (PyCFunction) interpolation, METH_VARARGS | METH_KEYWORDS,
         "Interpolation."},
    {"my_interp", (PyCFunction) my_interp, METH_VARARGS | METH_KEYWORDS,
         "my_interp."},         
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef interpolationmodule = {
    PyModuleDef_HEAD_INIT,
    "interpolation",
    NULL,
    -1,
    interpolation_methods
};

PyMODINIT_FUNC PyInit_interpolation(void)
{
    PyObject *mod = NULL;
    import_array();
    mod = PyModule_Create(&interpolationmodule);
    return mod;
}


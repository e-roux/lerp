
/*
PyObject* PyArray_SearchSorted(PyArrayObject* self, PyObject* values, NPY_SEARCHSIDE side, PyObject* perm)

https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.array.html#c.PyArray_SearchSorted

https://gist.github.com/saghul/1121260


Reference count: http://edcjones.tripod.com/refcount.html


 safe approach is to always use the generic
 operations  (functions whose name begins with 
 PyObject_, PyNumber_, PySequence_ or PyMapping_).
*/

// #define NPY_ALLOW_THREADS 1

#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL UTILS_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include "LERP_intern.h"
#include "NDTable.h"


// #define ARRAYD64(a) (PyArrayObject*) PyArray_FromAny(a, PyArray_DescrFromType(NPY_FLOAT64), 0, 0, 0, NULL)
#define ARRAYD64(a) (PyArrayObject*) PyArray_ContiguousFromAny(a, NPY_DOUBLE, 0, 0)


#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

#define DEBUG 0


PyObject *
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
    // PyObject_Print(fp, stdout, 0);
    // printf("\n");

    // printf("là\n");


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


Mesh_h Mesh_FromXarray(PyObject *mesh){

    // TODO : check that mesh is subclass of xarray
    // PyObject_IsSubclass(mesh , (PyObject*))
    /*    if (!PyArray_Check(data_array))
            goto out;
    */
    Mesh_h output = (Mesh_h) malloc(sizeof(Mesh_t));

    /**************************************************
    * Get paramters from mesh:
    *   - data
    *   - coords
    *   - dims
    **************************************************/
    // Return value: New reference.
    PyObject *data = PyObject_GetAttrString(mesh, "data");
    PyObject *coords = PyObject_GetAttrString(mesh, "coords");
    // PyObject *variable = PyObject_GetAttrString(mesh, "variable");
    PyObject *coords_list = PyObject_GetAttrString(mesh, "dims");

    // http://numerical.recipes/nr3_python_tutorial.html
    // // Cast data to npy_double
    // PyObject_Print(data, stdout, 0);

    // printf("\n");
    // printf("Refcount : %zi\n", data->ob_refcnt);
    // PyObject *array = PyArray_ContiguousFromAny(
    //                         data , NPY_DOUBLE, 1, 1);
   
    // printf("Refcount 1: %zi\n", data->ob_refcnt);
    PyArrayObject* array = ARRAYD64(data);
    // printf("Refcount 2: %zi\n", data->ob_refcnt);

    // printf("Refcount ARRAY 1 : %zi\n", PyArray_REFCOUNT(array));
    // printf("Refcount : %zi\n", data->ob_refcnt);

  
    // coords

    PyObject *key;

    // Check that data array has the same dim number as coords
    if (PySequence_Length(coords_list) != PyArray_NDIM(array)) {
        PyErr_SetString(PyExc_ValueError, 
            "Data and bkpts have different shapes");
        // todo : exit
    }

    npy_intp *shape_x = PyArray_SHAPE(array);
    output->ndim = PyArray_NDIM(array);
  
    for (Py_ssize_t j=0; j < output->ndim; j++) {
        output->shape[j] = *shape_x++;

        // printf("SHAPE %li\n", *--shape_x);

        /* TODO: check if is tuple */
        // if (PyTuple_Check(coords_list)) {
            /* PySequence_GetItem INCREFs key. */
        key = PyTuple_GetItem(coords_list, j);
        // }

        #if DEBUG == 1
        printf("Refcount key (1): %zi\n", key->ob_refcnt);
        #endif

        PyObject *axis = PyObject_GetAttrString(mesh,
            (char *)PyUnicode_AS_DATA(key));

        #if DEBUG == 1
        printf("Refcount key (2): %zi\n", key->ob_refcnt);
        #endif

        PyArrayObject *coords_tmp =  (PyArrayObject*) PyArray_ContiguousFromAny(
            axis, NPY_DOUBLE, 0, 0);
        output->coords[j] = PyArray_DATA(coords_tmp);

        Py_DECREF(axis);
        #if DEBUG == 1
        printf("Refcount key (3): %zi\n", key->ob_refcnt);
        #endif        
        // Py_DECREF(key);

    }
    output->data = PyArray_DATA(array);
    // printf("Refcount 2: %zi\n", data->ob_refcnt);

    output->size = PyArray_SIZE(array);
    output->itemsize = PyArray_ITEMSIZE(array);
    // output->interpmethod = &myfunction; // *interp_linear;
    // output->interpmethod = interpmethod; // *interp_linear;
    // output->extrapmethod = extrapmethod; // *interp_linear;


    Py_DECREF(coords_list);    
    Py_DECREF(coords);    
    // Py_DECREF(array);    
    // printf("Refcount ARRAY 2 : %zi\n", PyArray_REFCOUNT(array));
    // PyObject_Print(data, stdout, 0);
    Py_DECREF(data);
    Py_DECREF(data);

    /* Remianing references:
        data


    */ 

    return output;
}

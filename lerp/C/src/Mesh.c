
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
#include "NumPyWrapper.h"
#include "NDTable.h"
// #include "Mesh.h"

#define ARRAYD64(a) (PyArrayObject*) PyArray_ContiguousFromAny(a, NPY_DOUBLE, 0, 0)

#define DEBUG 0


PyObject *
my_interp(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{

    PyObject *fp, *xp, *x;
    PyObject *left = NULL, *right = NULL;
    PyArrayObject *fp_array = NULL, *xp_array = NULL, *x_array = NULL, *f_array = NULL;
    npy_intp i, x_len, xp_len;
    npy_double lval, rval;
    const npy_double *afp_data, *axp_data, *ax_data;
    npy_double *af_data, *slopes = NULL;

    static char *kwlist[] = {"x", "xp", "fp", "left", "right", NULL};

    NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "OOO|OO:my_interp", kwlist,
                                     &x, &xp, &fp, &left, &right)) {
        return NULL;
    }


    fp_array = (PyArrayObject *)PyArray_ContiguousFromAny(fp, NPY_DOUBLE, 1, 1);
    if (fp_array == NULL) {
        return NULL;
    }
    xp_array = (PyArrayObject *)PyArray_ContiguousFromAny(xp, NPY_DOUBLE, 1, 1);
    if (xp_array == NULL) {
        goto fail;
    }
    x_array = (PyArrayObject *)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 0, 0);
    if (x_array == NULL) {
        goto fail;
    }
    xp_len = PyArray_SIZE(xp_array);
    if (xp_len == 0) {
        PyErr_SetString(PyExc_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    if (PyArray_SIZE(fp_array) != xp_len) {
        PyErr_SetString(PyExc_ValueError,
                "fp and xp are not of the same length.");
        goto fail;
    }

    f_array = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(x_array),
                                                 PyArray_DIMS(x_array), NPY_DOUBLE);
    if (f_array == NULL) {
        goto fail;
    }
    x_len = PyArray_SIZE(x_array);

    afp_data = (const npy_double *)PyArray_DATA(fp_array);
    axp_data = (const npy_double *)PyArray_DATA(xp_array);
    
    ax_data = (const npy_double *)PyArray_DATA(x_array);
    af_data = (npy_double *)PyArray_DATA(f_array);
    
    lval = afp_data[0];
    rval = afp_data[xp_len - 1];

    /* binary_search_with_guess needs at least a 3 item long array */
    if (xp_len == 1) {
        const npy_double xp_val = axp_data[0];
        const npy_double fp_val = afp_data[0];

        NPY_BEGIN_THREADS_THRESHOLDED(x_len);
        for (i = 0; i < x_len; ++i) {
            const npy_double x_val = ax_data[i];
            af_data[i] = (x_val < xp_val) ? lval :
                                         ((x_val > xp_val) ? rval : fp_val);
        }
        NPY_END_THREADS;
    }
    else {
        npy_intp j = 0;

        /* only pre-calculate slopes if there are relatively few of them. */
        if (xp_len <= x_len) {
            slopes = PyArray_malloc((xp_len - 1) * sizeof(npy_double));
            if (slopes == NULL) {
                goto fail;
            }
        }

        NPY_BEGIN_THREADS;

        if (slopes != NULL) {
            for (i = 0; i < xp_len - 1; ++i) {
                slopes[i] = (afp_data[i+1] - afp_data[i]) / (axp_data[i+1] - axp_data[i]);
            }
        }

        for (i = 0; i < x_len; ++i) {
            const npy_double x_val = ax_data[i];

            if (npy_isnan(x_val)) {
                af_data[i] = x_val;
                continue;
            }

            j = binary_search_with_guess(x_val, axp_data, xp_len, j);  
            if (j == -1) {
                af_data[i] = lval;
            }
            else if (j == xp_len) {
                af_data[i] = rval;
            }
            else if (j == xp_len - 1) {
                af_data[i] = afp_data[j];
            }
            else {
                const npy_double slope = (slopes != NULL) ? slopes[j] :
                                         (afp_data[j+1] - afp_data[j]) / (axp_data[j+1] - axp_data[j]);
                af_data[i] = slope*(x_val - axp_data[j]) + afp_data[j];
            }
        }
        NPY_END_THREADS;
    }

    PyArray_free(slopes);
    Py_DECREF(fp_array);
    Py_DECREF(xp_array);
    Py_DECREF(x_array);
    return PyArray_Return(f_array);

    fail:
        Py_XDECREF(fp_array);
        Py_XDECREF(xp_array);
        Py_XDECREF(x_array);
        Py_XDECREF(f_array);
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
    output->array = ARRAYD64(data);
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
    Py_DECREF(array);    
    // printf("Refcount ARRAY 2 : %zi\n", PyArray_REFCOUNT(array));
    // PyObject_Print(data, stdout, 0);
    Py_DECREF(data);
    Py_DECREF(data);

    /* Remianing references:
        data


    */ 

    return output;
}

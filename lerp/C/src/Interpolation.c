
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
#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL UTILS_ARRAY_API
#include <numpy/arrayobject.h>

#include "NumPyWrapper.h"
#include "NDTable.h"
#include "Mesh.h"

#define ARRAYD64(arr) (PyArrayObject*) PyArray_ContiguousFromAny(arr, NPY_DOUBLE, 0, 0)
#define error_converting(x)  (((x) == -1) && PyErr_Occurred())
#define START_TIMING clock_t t; t = clock();
#define END_TIMING t = clock() - t; double time_taken = ((double)t)/CLOCKS_PER_SEC; printf("function took %f seconds to execute \n", time_taken);

#define DEBUG 0


NDTable_InterpMethod_t
get_interp_method(char *method)
{

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

NDTable_ExtrapMethod_t
get_extrap_method(char *method)
{

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

static PyObject
*interpolation(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict) 
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

    PyObject *ret = NULL;       // returned value, Py_BuildValue of result_array
    PyArrayObject *result_array = NULL;

    PyObject *mesh = NULL;      // function parameters from Python code
    PyObject *targets = NULL;   // function paramters from Python code

    npy_intp result_array_size;

    npy_intp      index[NPY_MAXDIMS]; // the subscripts
    npy_intp      nsubs[NPY_MAXDIMS]; // the neighboring subscripts
    npy_double    derivatives[NPY_MAXDIMS];
    npy_double    weigths[NPY_MAXDIMS]; // the weights for the interpolation
    npy_double    *params[NPY_MAXDIMS]; // the weights for the interpolation

    Mesh_h table;

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
    #if DEBUG == 1
    printf("Called successful\n");
    #endif

    /**************************************************
    * Check interpolation and extrapolation method
    **************************************************/
    NDTable_InterpMethod_t interpmethod = get_interp_method(interp_method);
    NDTable_ExtrapMethod_t extrapmethod = get_extrap_method(extrap_method);

    /**************************************************
    * Create Mesh_h
    **************************************************/


    // PyObject *data = PyObject_GetAttrString(mesh, "data");
    // PyArrayObject* array = (PyArrayObject*) PyArray_ContiguousFromAny(
    //                         data , NPY_DOUBLE, 0, 0);

    // printf("Refcount array : %zi\n", array);
    #if DEBUG == 1
    printf("Table creation...\n");
    #endif
    table = Mesh_FromXarray(mesh); // , *interpmethod, *extrapmethod);

    #if DEBUG == 1
    printf("... success.\n");
    #endif
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
        params[j] = PyArray_DATA(mytargets->coords[j]);

        // printf("%zd\n", PyArray_SIZE(mytargets->coords[j]));

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
    //    Py_DECREF(targets);

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
        // printf("ouais\n");
        result_data = &table->data[0];
        ret = Py_BuildValue("lf", result_data);
        goto out;
    }
    else {
        // TODO: add null check
       NPY_BEGIN_THREADS_DEF;
       NPY_BEGIN_THREADS_THRESHOLDED(result_array_size);
  
        // START_TIMING;
    
        // for(i = 0; i < params_size; i++) {
        //     // set array of parameter in each direction
        //     for(j = 0; j < mesh->ndim; j++) {
        //         // Find index and weights
        //         // NDTable_find_index(params[j][i], mesh->shape[j], mesh->coords[j],
        //         //                    &index[j], &weigths[j]);

        //         // index[j] = binary_search_with_guess(params[j][i], mesh->coords[j],
        //         //                                     mesh->shape[j], index[j]);

        //         _cache = binary_search_with_guess(params[j][i], mesh->coords[j],
        //                                           mesh->shape[j], _cache);
        //         index[j] = _cache;

        //         weigths[j] = (params[j][_cache] - mesh->coords[j][_cache]) /
        //                      (mesh->coords[j][_cache+1] - mesh->coords[j][_cache]);
        //     }
        //     npy_intp status = NDT_eval_internal(mesh, weigths, index, nsubs, 0, interp_method,
        //                                     extrap_method, &result[i], derivatives);

        //     if(status != NDTABLE_INTERPSTATUS_OK) {
        //         return -1;
        //     }
        // }

        // Iteration over each points
        for(i = 0; i < result_array_size; i++) {

            // for each point, iterate over each dimension
            // search index for interpolation and calculate weight.
            for(j = 0; j < table->ndim; j++) {
                // dx = (const npy_double *)PyArray_DATA(mytargets->coords[j]);

                // _cache will serve for next iteration as start value
                _cache = binary_search_with_guess(params[j][i],
                                                  table->coords[j],
                                                  table->shape[j],
                                                  _cache);

                /* Handle keys outside of the arr range first */
                if(_cache == -1) {
                    _cache = 0;
                 }
                else if(_cache == table->shape[j]) {
                    _cache = table->shape[j] - 2    ;
                }

                index[j] = _cache;
                weigths[j] = (params[j][i] - table->coords[j][_cache]) /
                             (table->coords[j][_cache+1] - table->coords[j][_cache]);

            // printf("%f between %f and %f, weight: %f\n", params[j][i],
            //               table->coords[j][_cache],
            //               table->coords[j][_cache+1],
            //               weigths[j]);
            }
            // PyObject_Print(PyList_GetItem(targets, 0), stdout, 0);
            #if DEBUG == 1
            printf("****************************************\n");
            #endif

            npy_intp status = NDT_eval_internal(
                table, weigths, index, nsubs, 0,
                interpmethod, extrapmethod, &result_data[i]);
 
            // result_data

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

    if (PyArray_SIZE(result_array) == 1) {
        // printf("%lf\n", result_data[0]);
        ret = Py_BuildValue("f", (double) result_data[0]);
        #if DEBUG == 1
        printf("RET:\n");
        PyObject_Print(ret, stdout, 0);
        printf("\n");
        #endif        
    }
    else{
        ret = Py_BuildValue("O", (PyObject *) result_array);
    }

    out:
        #if DEBUG == 1
        printf("Clean output\n");
        #endif
        return ret;
    fail:
        #if DEBUG == 1
        printf("Failed\n");
        #endif
        return NULL;        
}



static PyMethodDef interpolation_methods[] = {
    {"interpolation", (PyCFunction) interpolation,
     METH_VARARGS | METH_KEYWORDS, "Interpolation."},
    {"my_interp", (PyCFunction) my_interp,
     METH_VARARGS | METH_KEYWORDS, "my_interp."},         
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




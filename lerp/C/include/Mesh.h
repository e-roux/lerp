#ifndef MESH_H
#define MESH_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	npy_intp 	shape[NPY_MAXDIMS]; 	// Array of data array dimensions.
	npy_intp	ndim;			    	// Number of array dimensions.
	npy_intp	size;			    	// Number of elements in the array.
	npy_intp    itemsize;		    	// Length of one array element in bytes.
	npy_double  *data;			    	// Buffer object pointing to the start
										// of the array’s data.
	npy_double  *coords[NPY_MAXDIMS]; 	// array of pointers to the coords values
	// npy_intp    (*interpmethod)(npy_intp);		    // Function for interpolation
} Mesh_t;

typedef Mesh_t * Mesh_h;


PyObject * my_interp(PyObject *, PyObject *, PyObject *);

Mesh_h Mesh_FromXarray(PyObject *);


#ifdef __cplusplus
}
#endif

#endif
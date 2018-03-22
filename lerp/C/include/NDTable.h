/*
BSD 3-Clause License

Copyright (c) 2017, Dassault Systemes.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "numpy/npy_math.h"
#include <time.h>

#define START_TIMING clock_t t; t = clock();
#define END_TIMING t = clock() - t; double time_taken = ((double)t)/CLOCKS_PER_SEC; printf("function took %f seconds to execute \n", time_taken);


#ifndef NDTABLE_H_
#define NDTABLE_H_

#ifdef __cplusplus
extern "C" {
#endif


/*! Interpolation methods */
/*typedef enum {
	interp_hold = 1,
	interp_nearest,
	interp_linear,
	interp_akima,
	interp_fritsch_butland,
	interp_steffen
} MESH_INTERP_FUNCTION;
*/

/*! Interpolation methods */
typedef enum {
	NDTABLE_INTERP_HOLD = 1,
	NDTABLE_INTERP_NEAREST,
	NDTABLE_INTERP_LINEAR,
	NDTABLE_INTERP_AKIMA,
	NDTABLE_INTERP_FRITSCH_BUTLAND,
	NDTABLE_INTERP_STEFFEN
} NDTable_InterpMethod_t;

/*! Extrapolation methods */
typedef enum {
    NDTABLE_EXTRAP_HOLD = 1,
	NDTABLE_EXTRAP_LINEAR,
	NDTABLE_EXTRAP_NONE
} NDTable_ExtrapMethod_t;


/* Array attributes */
typedef struct {
	npy_intp 	shape[NPY_MAXDIMS];   // Array of data array dimensions.
	npy_intp		ndim;			    // Number of array dimensions.
	npy_double *data;			    // Buffer object pointing to the start
								// of the array’s data.
	npy_intp		size;			    // Number of elements in the array.
	npy_intp     itemsize;		    // Length of one array element in bytes.
	npy_double *coords[NPY_MAXDIMS]; //!< array of pointers to the scale values
	npy_intp     (*interpmethod)(npy_intp);		    // Function for interpolation
} NDTable_t;

typedef NDTable_t * NDTable_h;

/*! Interpolation status codes */
typedef enum {
	NDTABLE_INTERPSTATUS_UNKNOWN_METHOD  = -4,
	NDTABLE_INTERPSTATUS_DATASETNOTFOUND = -3,
	NDTABLE_INTERPSTATUS_WRONGNPARAMS    = -2,
	NDTABLE_INTERPSTATUS_OUTOFBOUNS      = -1,
    NDTABLE_INTERPSTATUS_OK              =  0
} NDTable_InterpolationStatus;


/*! Sets the error message */
void NDTable_set_error_message(const char *msg, ...);


npy_double NDTable_get_value_subs(const NDTable_h table, const npy_intp subs[]);




npy_intp NDT_eval_internal(const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double *derivatives);


/*! Evalute the total differential of the table at the given sample point and deltas using the specified inter- and extrapolation methods
 *
 * @param [in]	table			the table handle
 * @param [in]	nparams			the number of dimensions
 * @param [in]	params			the sample point
 * @param [in]	delta_params	the the deltas
 * @param [in]	interp_method	the interpolation method
 * @param [in]	extrap_method	the extrapolation method
 * @param [out]	value			the total differential at the sample point
 *
 * @return		0 if the value could be evaluated, -1 otherwise
 */

typedef npy_intp(*interp_fun)(const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, 
						 NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, 
						 npy_double *value, npy_double derivatives[]);

// forward declare inter- and extrapolation functions
static npy_intp interp_hold		      (const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double derivatives[]);
static npy_intp interp_nearest	      (const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double derivatives[]);
static npy_intp interp_linear	      (const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double derivatives[]);
static npy_intp interp_akima		      (const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double derivatives[]);
static npy_intp interp_fritsch_butland (const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double derivatives[]);
static npy_intp interp_steffen         (const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double derivatives[]);
static npy_intp extrap_hold		      (const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double derivatives[]);
static npy_intp extrap_linear	      (const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double derivatives[]);



// static npy_intp
// binary_search_with_guess(const npy_double key, const npy_double *arr,
//                          npy_intp len, npy_intp guess);


#ifdef __cplusplus
}
#endif

#endif /*NDTABLE_H_*/

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

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
// #include <stdio.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include "Mesh.h"


#ifndef NDTABLE_H_
#define NDTABLE_H_

#ifdef __cplusplus
extern "C" {
#endif


/*
Macro for interpolation function definition

Paramters
---------

table			:	Mesh_h
					Table handle
weight			:	npy_double
					Weights for the interpolation (normalized)
subs    		:   npy_double
					Subscripts of the left sample point
nsubs			:   npy_double
					Subscripts of the right (next) sample point
dim				:	npy_intp
					Index of the current dimension
interp_method	:	NDTable_InterpMethod_t
					Interpolation method
extrap_method	:	NDTable_ExtrapMethod_t
					Extrapolation method
result			: 	npy_double
					interpolated result
*/
#define INTERP_PARAMETERS (const Mesh_h, const npy_double *, const npy_intp *, npy_intp *, npy_intp, NDTable_InterpMethod_t, NDTable_ExtrapMethod_t, npy_double *);

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
	npy_intp 	shape[NPY_MAXDIMS]; 	  // Array of data array dimensions.
	npy_intp	ndim;			    	 // Number of array dimensions.
	PyArrayObject  *coords[NPY_MAXDIMS]; //!< array of pointers to the scale values
} NDTargets_t;

/* Array attributes */
typedef struct {
	npy_intp   size;				// Number of elements in the array.
	npy_double *data;			    // Buffer object pointing to the start
} NDResult_t;

typedef NDTargets_t * NDTargets_h;
typedef NDResult_t  * NDResult_h;

/*! Interpolation status codes */
typedef enum {
	NDTABLE_INTERPSTATUS_UNKNOWN_METHOD  = -4,
	NDTABLE_INTERPSTATUS_DATASETNOTFOUND = -3,
	NDTABLE_INTERPSTATUS_WRONGNPARAMS    = -2,
	NDTABLE_INTERPSTATUS_OUTOFBOUNS      = -1,
    NDTABLE_INTERPSTATUS_OK              =  0
} NDTable_InterpolationStatus;


npy_intp NDT_eval_internal INTERP_PARAMETERS;

typedef npy_intp ( *interp_fun ) INTERP_PARAMETERS;

static npy_intp interp_hold INTERP_PARAMETERS;
static npy_intp interp_nearest INTERP_PARAMETERS;
static npy_intp interp_linear INTERP_PARAMETERS;
static npy_intp interp_akima INTERP_PARAMETERS;
static npy_intp interp_fritsch_butland INTERP_PARAMETERS;
static npy_intp interp_steffen INTERP_PARAMETERS;
static npy_intp extrap_hold INTERP_PARAMETERS;
static npy_intp extrap_linear INTERP_PARAMETERS;

#ifdef __cplusplus
}
#endif

#endif

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
#include "NDTable.h"
#include <numpy/npy_math.h>


#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
#endif
	

#ifndef NAN
static const unsigned long __nan[2] = { 0xffffffff, 0x7fffffff };
#define NAN (*(const float *) __nan)
#endif

#ifdef _WIN32
#define ISFINITE(x) _finite(x)
#else
#define ISFINITE(x) isfinite(x)
#endif

#define DEBUG 2	


/**

Parameters
----------

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

Returns
-------
status code
*/

npy_intp NDT_eval_internal(const Mesh_h table, const npy_double *weigths,
						   const npy_intp *subs, npy_intp *nsubs, npy_intp dim,
	     				   NDTable_InterpMethod_t interp_method,
	     				   NDTable_ExtrapMethod_t extrap_method,
	     				   npy_double *result)
{
	interp_fun func;


	// check arguments
	if (weigths == NULL || subs == NULL ||
		nsubs == NULL ||  result == NULL ) {
		return -1;
	}

	#if DEBUG == 1
	printf("Dans NDT_eval_internal, dim =: %li/%li\n", dim, table->ndim);
	#endif

	if (dim >= table->ndim) {
		npy_intp index = 0;
		npy_intp k = 1;
		for(npy_intp i = table->ndim-1; i >= 0; i--) {
			index += nsubs[i] * k; 
			k *= table->shape[i]; // TODO use pre-calculated offsets
		}	
		*result =  table->data[index];
		return 0;
	}

	// find the right function:
	if(table->shape[dim] < 2) {
		func = interp_hold;
	} else if (weigths[dim] < 0.0 || weigths[dim] > 1.0) {
		// extrapolate
		switch (extrap_method) {
			case NDTABLE_EXTRAP_HOLD:
				func = extrap_hold;
				break;
			case NDTABLE_EXTRAP_LINEAR:
				switch (interp_method) {
				case NDTABLE_INTERP_AKIMA:           func = interp_akima;           break;
				case NDTABLE_INTERP_FRITSCH_BUTLAND: func = interp_fritsch_butland; break;
				default:                             func = extrap_linear;			break;
		}
		break;

		default:
			printf("Requested value is outside data range");
			return -1;
		}
	} else {
		// interpolate
		switch (interp_method) {
		case NDTABLE_INTERP_HOLD:	         func = interp_hold;            break;
		case NDTABLE_INTERP_NEAREST:         func = interp_nearest;         break;
		case NDTABLE_INTERP_LINEAR:          func = interp_linear;          break;
		case NDTABLE_INTERP_AKIMA:			 func = interp_akima;           break;
		case NDTABLE_INTERP_FRITSCH_BUTLAND: func = interp_fritsch_butland; break;
		case NDTABLE_INTERP_STEFFEN:         func = interp_steffen;         break;
		default: return -1; // TODO: set error message
		}
	}

	return (*func)(table, weigths, subs, nsubs, dim, interp_method, extrap_method, result);
}

static npy_intp interp_hold(const Mesh_h table, const npy_double *weight, const npy_intp *subs,
							npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method,
							NDTable_ExtrapMethod_t extrap_method, npy_double *result)
{
	nsubs[dim] = subs[dim]; // always take the left sample value

	return NDT_eval_internal(table, weight, subs, nsubs, dim + 1, interp_method, extrap_method, result);
}

static npy_intp interp_nearest(const Mesh_h table, const npy_double *weight, const npy_intp *subs,
							   npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method,
							   NDTable_ExtrapMethod_t extrap_method, npy_double *result)
{
	npy_intp err;
	nsubs[dim] = weight[dim] < 0.5 ? subs[dim] : subs[dim] + 1;

	if ((err = NDT_eval_internal(table, weight, subs, nsubs, dim + 1,
								 interp_method, extrap_method,
								 result)) != 0) {
		return err;
	}

	// if the value is not finite return NAN
	if (!ISFINITE(*result)) {
		*result = NAN;
	}

	return 0;
}

static npy_intp interp_linear(const Mesh_h table, const npy_double *weight,
							  const npy_intp *subs, npy_intp *nsubs, npy_intp dim,
							  NDTable_InterpMethod_t interp_method,
							  NDTable_ExtrapMethod_t extrap_method,
							  npy_double *result) 
{
	npy_intp err;
	npy_double a, b;

	// get the left value
	nsubs[dim] = subs[dim];


	#if DEBUG == 1
	printf("Dans interp_linear, dim =: %li\n", dim);
	#endif

	if ((err = NDT_eval_internal(table, weight, subs, nsubs, dim + 1, 
								 interp_method, extrap_method,
								 &a)) != 0) {
			#if DEBUG == 1
			printf("Dans interp_linear (1)), dim =: %li\n", dim);
			#endif
			return err;
	}

	// get the right value
	nsubs[dim] = subs[dim] + 1;

	if ((err = NDT_eval_internal(table, weight, subs, nsubs, dim + 1,
								 interp_method, extrap_method,
								 &b)) != 0) {
			#if DEBUG == 1
			printf("Dans interp_linear (1)), dim =: %li\n", dim);
			#endif
			return err;
	}

	// if any of the values is not finite return NAN
	if (npy_isnan(a) || npy_isnan(b)) {
			#if DEBUG == 2
			printf("Dans interp_linear (NaN)), dim =: %li, a=%f, b=%f\n", dim, a, b);
			#endif
			*result = NAN;
		return 0;
	}

	#if DEBUG == 1
	printf("Dans interp_linear (Finish)), dim =: %li\n", dim);
	#endif
	// calculate the interpolated value
	*result = (1 - weight[dim]) * a + weight[dim] * b;




	return 0;
}

static void cubic_hermite_spline(const npy_double x0, const npy_double x1,
								 const npy_double y0, const npy_double y1,
								 const npy_double weight, const npy_double c[4],
								 npy_double *result)
{

	npy_double v;

	if (weight < 0) { // extrapolate left
		*result = y0 + c[2] * ((x1 - x0) * weight);
	} else if (weight <= 1) { // interpolate
		v = (x1 - x0) * weight;
		*result = ((c[0] * v + c[1]) * v + c[2]) * v + c[3];
	} else { // extrapolate right
		v = x1 - x0;
		*result = y1 + ((3 * c[0] * v + 2 * c[1]) * v + c[2]) * (v * (weight - 1));
	}
}

static npy_intp interp_akima(const Mesh_h table, const npy_double *weight,
							 const npy_intp *subs, npy_intp *nsubs, npy_intp dim,
							 NDTable_InterpMethod_t interp_method,
							 NDTable_ExtrapMethod_t extrap_method,
							 npy_double *result) 
{

	npy_double x[6] = { 0, 0, 0, 0, 0, 0};
	npy_double y[6] = { 0, 0, 0, 0, 0, 0};
	npy_double c[4] = { 0, 0, 0, 0 };	   // spline coefficients
    npy_double d[5] = { 0, 0, 0, 0, 0 };   // divided differences
    npy_double c2   = 0;
	npy_double dx   = 0;
	npy_double a    = 0;
	// npy_double v    = 0;

	npy_intp n = table->shape[dim]; // extent of the current dimension
	npy_intp sub = subs[dim];      // subscript of current dimension
	npy_intp err, i, idx;

	for (i = 0; i < 6; i++) {
		idx = sub - 2 + i;

		if (idx >= 0 && idx < n) {
			x[i] = table->coords[dim][idx];

			nsubs[dim] = idx;
			if ((err = NDT_eval_internal(table, weight, subs, nsubs, dim + 1,
										 interp_method, extrap_method,
										 &y[i])) != 0) {
				return err;
			}
		}
	}

	// if any of the values is not finite return NAN
	for (i = 0; i < 6; i++) {
		if (!ISFINITE(y[i])) {
			*result = NAN;
			return 0;
		}
	}

	// calculate the divided differences
	for (i = PyArray_MAX(0, 2 - sub); i < PyArray_MIN(5, 1 + n - sub); i++) {
		d[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
	}

	// pad left
	if (sub < 2) {
		if (sub < 1) {
			d[1] = 2.0 * d[2] - d[3];
		}
		d[0] = 2.0 * d[1] - d[2];
	}

	// pad right
	if (sub > n - 4) {
		if (sub > n - 3) {
			d[3] = 2.0 * d[2] - d[1];
		}
		d[4] = 2.0 * d[3] - d[2];
	}

    // initialize the left boundary slope
    c2 = fabs(d[3] - d[2]) + fabs(d[1] - d[0]);

	if (c2 > 0) {
        a = fabs(d[1] - d[0]) / c2;
        c2 = (1 - a) * d[1] + a * d[2];
    } else {
        c2 = 0.5 * d[1] + 0.5 * d[2];
    }

    // calculate the coefficients
	dx = x[3] - x[2];

    c[2] = c2;
    c2 = fabs(d[4] - d[3]) + fabs(d[2] - d[1]);

	if (c2 > 0) {
        a = fabs(d[2] - d[1]) / c2;
        c2 = (1 - a) * d[2] + a * d[3];
    } else {
        c2 = 0.5 * d[2] + 0.5 * d[3];
    }

	c[1] = (3 * d[2] - 2 * c[2] - c2) / dx;
	c[0] = (c[2] + c2 - 2 * d[2]) / (dx * dx);

	c[3] = y[2];

	cubic_hermite_spline(x[2], x[3], y[2], y[3], weight[dim], c, result);

	return 0;
}

static npy_intp interp_fritsch_butland(const Mesh_h table, const npy_double *weight,
									   const npy_intp *subs, npy_intp *nsubs, npy_intp dim,
									   NDTable_InterpMethod_t interp_method,
									   NDTable_ExtrapMethod_t extrap_method,
									   npy_double *result)
{
	npy_double x [4] = { 0, 0, 0, 0 };
	npy_double y [4] = { 0, 0, 0, 0 };
	npy_double dx[3] = { 0, 0, 0 };
	npy_double d [3] = { 0, 0, 0 };    // divided differences
    npy_double c [4] = { 0, 0, 0, 0 }; // spline coefficients
    npy_double c2    = 0;

	npy_intp n = table->shape[dim]; // extent of the current dimension
	npy_intp sub = subs[dim];      // subscript of current dimension
	npy_intp err, i, idx;

	for (i = 0; i < 4; i++) {
		idx = sub - 1 + i;

		if (idx >= 0 && idx < n) {
			x[i] = table->coords[dim][idx];

			nsubs[dim] = idx;
			if ((err = NDT_eval_internal(table, weight, subs, nsubs, dim + 1,
										 interp_method, extrap_method, &y[i])) != 0) {
				return err;
			}
		}
	}

	// if any of the values is not finite return NAN
	for (i = 0; i < 4; i++) {
		if (!ISFINITE(y[i])) {
			*result = NAN;
			return 0;
		}
	}

	// calculate the divided differences
	//for (i = MAX(0, 1 - sub); i < MIN(3, n - 1 - sub); i++) {
	for (i = 0; i < 3; i++) {
		dx[i] = x[i + 1] - x[i];
		d[i] = (y[i + 1] - y[i]) / dx[i];
	}

    // initialize the left boundary slope

    // calculate the coefficients

	if (sub == 0) {
        c2 = d[1];
    } else if (d[0] == 0 || d[1] == 0 || (d[0] < 0 && d[1] > 0) || (d[0] > 0 && d[1] < 0)) {
        c2 = 0;
     } else {
 		c2 = 3 * (dx[0] + dx[1]) / ((dx[0] + 2 * dx[1]) / d[0] + (dx[1] + 2 * dx[0]) / d[1]);
 	}

    c[2] = c2;

    if (sub == n - 2) {
        c2 = d[1];
    } else if (d[1] == 0 || d[2] == 0 || (d[1] < 0 && d[2] > 0) || (d[1] > 0 && d[2] < 0)) {
        c2 = 0;
     } else {
 		c2 = 3 * (dx[1] + dx[2]) / ((dx[1] + 2 * dx[2]) / d[1] + (dx[2] + 2 * dx[1]) / d[2]);
 	}

    c[1] = (3 * d[1] - 2 * c[2] - c2) / dx[1];
    c[0] = (c[2] + c2 - 2 * d[1]) / (dx[1] * dx[1]);

    c[3] = y[1];

	cubic_hermite_spline(x[1], x[2], y[1], y[2], weight[dim], c, result);

	return 0;
}

static npy_intp interp_steffen(const Mesh_h table, const npy_double *weight,
							   const npy_intp *subs, npy_intp *nsubs, npy_intp dim,
							   NDTable_InterpMethod_t interp_method,
							   NDTable_ExtrapMethod_t extrap_method,
							   npy_double *result)
{
	npy_double x [4] = { 0, 0, 0, 0 };
	npy_double y [4] = { 0, 0, 0, 0 };
	npy_double dx[3] = { 0, 0, 0 };
	npy_double d [3] = { 0, 0, 0 };    // divided differences
    npy_double c [4] = { 0, 0, 0, 0 }; // spline coefficients
    npy_double c2    = 0;

	const npy_intp n   = table->shape[dim]; // extent of the current dimension
	const npy_intp sub = subs[dim];      // subscript of current dimension
	npy_intp err, i, idx;

	for (i = 0; i < 4; i++) {
		idx = sub - 1 + i;

		if (idx >= 0 && idx < n) {
			x[i] = table->coords[dim][idx];

			nsubs[dim] = idx;
			if ((err = NDT_eval_internal(table, weight, subs, nsubs, dim + 1,
										 interp_method, extrap_method, &y[i])) != 0) {
				return err;
			}
		}
	}

	// if any of the values is not finite return NAN
	for (i = 0; i < 4; i++) {
		if (!ISFINITE(y[i])) {
			*result = NAN;
			return 0;
		}
	}

	// calculate the divided differences
	for (i = 0; i < 3; i++) {
		dx[i] = x[i + 1] - x[i];
		d[i] = (y[i + 1] - y[i]) / dx[i];
	}

	// calculate the coefficients
	if (sub == 0) {
        c2 = d[1];
    } else if (d[0] == 0 || d[1] == 0 || (d[0] < 0 && d[1] > 0) || (d[0] > 0 && d[1] < 0)) {
        c2 = 0;
    } else {
        npy_double half_abs_c2, abs_di, abs_di1;
        c2 = (d[0] * dx[1] + d[1] * dx[0]) / (dx[0] + dx[1]);
        half_abs_c2 = 0.5 * fabs(c2);
        abs_di = fabs(d[0]);
        abs_di1 = fabs(d[1]);
        if (half_abs_c2 > abs_di || half_abs_c2 > abs_di1) {
            const npy_double two_a = d[0] > 0 ? 2 : -2;
            c2 = two_a*(abs_di < abs_di1 ? abs_di : abs_di1);
        }
    }

    c[2] = c2;

	if (sub == n - 2) {
        c2 = d[1];
    } else if (d[1] == 0 || d[2] == 0 || (d[1] < 0 && d[2] > 0) || (d[1] > 0 && d[2] < 0)) {
        c2 = 0;
    } else {
        npy_double half_abs_c2, abs_di, abs_di1;
        c2 = (d[1] * dx[2] + d[2] * dx[1]) / (dx[1] + dx[2]);
        half_abs_c2 = 0.5 * fabs(c2);
        abs_di = fabs(d[1]);
        abs_di1 = fabs(d[2]);
        if (half_abs_c2 > abs_di || half_abs_c2 > abs_di1) {
            const npy_double two_a = d[1] > 0 ? 2 : -2;
            c2 = two_a*(abs_di < abs_di1 ? abs_di : abs_di1);
        }
    }

    c[1] = (3 * d[1] - 2 * c[2] - c2) / dx[1];
    c[0] = (c[2] + c2 - 2 * d[1]) / (dx[1] * dx[1]);
    c[3] = y[1];

	cubic_hermite_spline(x[1], x[2], y[1], y[2], weight[dim], c, result);

	return 0;
}


static npy_intp extrap_hold(const Mesh_h table, const npy_double *weigths,
							const npy_intp *subs, npy_intp *nsubs, npy_intp dim,
							NDTable_InterpMethod_t interp_method,
							NDTable_ExtrapMethod_t extrap_method,
							npy_double *result)
{
	npy_intp err;
	nsubs[dim] = weigths[dim] < 0.0 ? subs[dim] : subs[dim] + 1;

	if ((err = NDT_eval_internal(table, weigths, subs, nsubs, dim + 1,
								 interp_method, extrap_method, result)) != 0) {
		return err;
	}

	// if the value is not finite return NAN
	if (!ISFINITE(*result)) {
		*result = NAN;
	}

	return 0;
}


static npy_intp extrap_linear(const Mesh_h table, const npy_double *weigths,
							  const npy_intp *subs, npy_intp *nsubs, npy_intp dim,
							  NDTable_InterpMethod_t interp_method,
							  NDTable_ExtrapMethod_t extrap_method,
							  npy_double *result)
{
	npy_intp err;
	npy_double a, b;

	nsubs[dim] = subs[dim];
	if ((err = NDT_eval_internal(table, weigths, subs, nsubs, dim + 1,
								 interp_method, extrap_method, &a)) != 0) {
		return err;
	}

	nsubs[dim] = subs[dim] + 1;
	if ((err = NDT_eval_internal(table, weigths, subs, nsubs, dim + 1,
								 interp_method, extrap_method, &b)) != 0) {
		return err;
	}

	// if any of the values is not finite return NAN
	if (!ISFINITE(a) || !ISFINITE(b)) {
		*result = NAN;
		return 0;
	}

	// calculate the extrapolated value
	*result = (1 - weigths[dim]) * a + weigths[dim] * b;

	#if DEBUG == 1
	printf("Dans extrap_linear : %f\n", *result);
	#endif

	return 0;
}

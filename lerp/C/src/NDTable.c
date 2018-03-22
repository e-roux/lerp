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

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <NDTable.h>


#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
// https://docs.python.org/3/faq/windows.html#is-a-pyd-file-the-same-as-a-dll
void PyInit_libNDTable() {}
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


#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif


/**
Prototype of an interpolation function

@param  table			[in]		table handle
@param  t				[in]		weights for the interpolation (normalized)
@param  subs			[in]		subscripts of the left sample point
@param  nsubs			[in,out]	subscripts of the right (next) sample point
@param  dim				[in]		index of the current dimension
@param  interp_method	[in]		index of the current dimension
@param  extrap_method	[in]		index of the current dimension
@param  value			[out]		interpoated value
@param  derivatives		[out]		partial derivatives

@return status code
*/
npy_intp NDT_eval_internal(const NDTable_h table, const npy_double *weigths,
						   const npy_intp *subs, npy_intp *nsubs, npy_intp dim,
	     				   NDTable_InterpMethod_t interp_method,
	     				   NDTable_ExtrapMethod_t extrap_method,
	     				   npy_double *value, npy_double derivatives[])
{
	npy_intp i, k = 1;
	npy_intp index;
	interp_fun func;

	// check arguments
	if (weigths == NULL || subs == NULL || nsubs == NULL || 
		value == NULL || derivatives == NULL) {
		return -1;
	}

	if (dim >= table->ndim) {
		index = 0;

		for(i = table->ndim-1; i >= 0; i--) {
			index += subs[i] * k;
			k *= table->shape[i]; // TODO use pre-calculated offsets
		}

		*value = table->data[index];

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
			sprintf("", "Requested value is outside data range");
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

	return (*func)(table, weigths, subs, nsubs, dim, interp_method, extrap_method, value, derivatives);
}

static npy_intp interp_hold(const NDTable_h table, const npy_double *t, const npy_intp *subs,
							npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method,
							NDTable_ExtrapMethod_t extrap_method, npy_double *value,
							npy_double slopes[]) {
	nsubs[dim] = subs[dim]; // always take the left sample value

	slopes[dim] = 0;
	return NDT_eval_internal(table, t, subs, nsubs, dim + 1, interp_method, extrap_method, value, slopes);
}

static npy_intp interp_nearest(const NDTable_h table, const npy_double *t, const npy_intp *subs,
							   npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method,
							   NDTable_ExtrapMethod_t extrap_method, npy_double *value,
							   npy_double slopes[]) {
	npy_intp err;
	nsubs[dim] = t[dim] < 0.5 ? subs[dim] : subs[dim] + 1;
	slopes[dim] = 0;

	if ((err = NDT_eval_internal(table, t, subs, nsubs, dim + 1, interp_method, extrap_method, value, slopes)) != 0) {
		return err;
	}

	// if the value is not finite return NAN
	if (!ISFINITE(*value)) {
		*value = NAN;
		slopes[dim] = NAN;
	}

	return 0;
}

static npy_intp interp_linear(const NDTable_h table, const npy_double *t,
							  const npy_intp *subs, npy_intp *nsubs, npy_intp dim,
							  NDTable_InterpMethod_t interp_method,
							  NDTable_ExtrapMethod_t extrap_method,
							  npy_double *value, npy_double slopes[])
{
	npy_intp err;
	npy_double a, b;

	// get the left value
	nsubs[dim] = subs[dim];

	if ((err = NDT_eval_internal(table, t, subs, nsubs, dim + 1, 
								 interp_method, extrap_method,
								 &a, slopes)) != 0) {
		return err;
	}

	// get the right value
	nsubs[dim] = subs[dim] + 1;

	if ((err = NDT_eval_internal(table, t, subs, nsubs, dim + 1,
								 interp_method, extrap_method,
								 &b, slopes)) != 0) {
		return err;
	}

	// if any of the values is not finite return NAN
	if (npy_isnan(a) || !npy_isnan(b)) {
		*value = NAN;
		slopes[dim] = NAN;
		return 0;
	}

	// calculate the interpolated value
	*value = (1 - t[dim]) * a + t[dim] * b;

	// calculate the derivative
	slopes[dim] = (b - a) / (table->coords[dim][subs[dim] + 1] - table->coords[dim][subs[dim]]);

	return 0;
}

static void cubic_hermite_spline(const npy_double x0, const npy_double x1, const npy_double y0, const npy_double y1, const npy_double t, const npy_double c[4], npy_double *value, npy_double *derivative) {

	npy_double v;

	if (t < 0) { // extrapolate left
		*value = y0 + c[2] * ((x1 - x0) * t);
		*derivative = c[2];
	} else if (t <= 1) { // interpolate
		v = (x1 - x0) * t;
		*value = ((c[0] * v + c[1]) * v + c[2]) * v + c[3];
		*derivative = (3 * c[0] * v + (2 * c[1])) * v + c[2];
	} else { // extrapolate right
		v = x1 - x0;
		*value = y1 +   ((3 * c[0] * v + 2 * c[1]) * v + c[2]) * (v * (t - 1));
		*derivative = (3 * c[0] * v + 2 * c[1]) * v + c[2];
	}
}

static npy_intp interp_akima(const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double slopes[]) {

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
			if ((err = NDT_eval_internal(table, t, subs, nsubs, dim + 1, interp_method, extrap_method, &y[i], slopes)) != 0) {
				return err;
			}
		}
	}

	// if any of the values is not finite return NAN
	for (i = 0; i < 6; i++) {
		if (!ISFINITE(y[i])) {
			*value = NAN;
			slopes[dim] = NAN;
			return 0;
		}
	}

	// calculate the divided differences
	for (i = MAX(0, 2 - sub); i < MIN(5, 1 + n - sub); i++) {
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

	cubic_hermite_spline(x[2], x[3], y[2], y[3], t[dim], c, value, &slopes[dim]);

	return 0;
}

static npy_intp interp_fritsch_butland(const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double slopes[]) {

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
			if ((err = NDT_eval_internal(table, t, subs, nsubs, dim + 1, interp_method, extrap_method, &y[i], slopes)) != 0) {
				return err;
			}
		}
	}

	// if any of the values is not finite return NAN
	for (i = 0; i < 4; i++) {
		if (!ISFINITE(y[i])) {
			*value = NAN;
			slopes[dim] = NAN;
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

	cubic_hermite_spline(x[1], x[2], y[1], y[2], t[dim], c, value, &slopes[dim]);

	return 0;
}

static npy_intp interp_steffen(const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double slopes[]) {

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
			if ((err = NDT_eval_internal(table, t, subs, nsubs, dim + 1, interp_method, extrap_method, &y[i], slopes)) != 0) {
				return err;
			}
		}
	}

	// if any of the values is not finite return NAN
	for (i = 0; i < 4; i++) {
		if (!ISFINITE(y[i])) {
			*value = NAN;
			slopes[dim] = NAN;
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

	cubic_hermite_spline(x[1], x[2], y[1], y[2], t[dim], c, value, &slopes[dim]);

	return 0;
}

static npy_intp extrap_hold(const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double slopes[]) {
	npy_intp err;
	nsubs[dim] = t[dim] < 0.0 ? subs[dim] : subs[dim] + 1;
	slopes[dim] = 0;

	if ((err = NDT_eval_internal(table, t, subs, nsubs, dim + 1, interp_method, extrap_method, value, slopes)) != 0) {
		return err;
	}

	// if the value is not finite return NAN
	if (!ISFINITE(*value)) {
		*value = NAN;
		slopes[dim] = NAN;
	}

	return 0;
}

static npy_intp extrap_linear(const NDTable_h table, const npy_double *t, const npy_intp *subs, npy_intp *nsubs, npy_intp dim, NDTable_InterpMethod_t interp_method, NDTable_ExtrapMethod_t extrap_method, npy_double *value, npy_double slopes[]) {
	npy_intp err;
	npy_double a, b;

	nsubs[dim] = subs[dim];
	if ((err = NDT_eval_internal(table, t, subs, nsubs, dim + 1, interp_method, extrap_method, &a, slopes)) != 0) {
		return err;
	}

	nsubs[dim] = subs[dim] + 1;
	if ((err = NDT_eval_internal(table, t, subs, nsubs, dim + 1, interp_method, extrap_method, &b, slopes)) != 0) {
		return err;
	}

	// if any of the values is not finite return NAN
	if (!ISFINITE(a) || !ISFINITE(b)) {
		*value = NAN;
		slopes[dim] = NAN;
		return 0;
	}

	// calculate the extrapolated value
	*value = (1 - t[dim]) * a + t[dim] * b;

	// calculate the derivative
	slopes[dim] = (b - a) / (table->coords[dim][subs[dim] + 1] - table->coords[dim][subs[dim]]);

	return 0;
}





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

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include <stdarg.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdio.h>

#include "NDTable.h"

#ifdef _WIN32
#define ISFINITE(x) _finite(x)
/*
https://docs.python.org/3/faq/windows.html#is-a-pyd-file-the-same-as-a-dll
*/
void PyInit_libNDTable() {}
#else
#define ISFINITE(x) isfinite(x)
#endif

static char error_message[MAX_MESSAGE_LENGTH] = "";


void NDTable_set_error_message(const char *msg, ...) {
	va_list vargs;
	va_start(vargs, msg);
	vsprintf(error_message, msg, vargs);
	va_end(vargs);
}

NDTable_h NDTable_alloc_table() {
	return (NDTable_h )calloc(1, sizeof(NDTable_t));
}

void NDTable_free_table(NDTable_h table) {
	if(!table) return;

	free(table);
}

const char * NDTable_get_error_message() {
	return error_message;
}

void NDTable_calculate_offsets(int ndim, const int shape[], int *strides) {
	int i;

	if(ndim < 1) {
		return;
	}

	strides[ndim-1] = 1;

	for(i = ndim-2; i >= 0; i--) {
		strides[i] = strides[i+1] * shape[i+1];
	}
}

int NDTable_validate_table(NDTable_h table) {
	int i, j, size, strides[MAX_NDIMS];
	double v;

	// check the rank
	if(table->ndim < 0 || table->ndim > 32) {
		NDTable_set_error_message("The rank of '%s' in '%s' must be in the range [0;32] but was %d", "table->datasetname", "table->filename", table->ndim);
		return -1;
	}

	// check the extent of the dimensions
	for(i = 0; i < table->ndim; i++) {
		if(table->shape[i] < 1) {
			NDTable_set_error_message("Extent of dimension %d of '%s' in '%s' must be >=0 but was %d", i, "table->datasetname", "table->filename", table->shape[i]);
			return -1;
		}
	}

	// check the number of values
	size = 1;
	for(i = 0; i < table->ndim; i++) {
		size = size * table->shape[i];
	}

	if(table->size != size) {
		NDTable_set_error_message("The size of '%s' in '%s' does not match its extent", "table->datasetname", "table->filename");
		return -1;
	}

	// check the offsets
	NDTable_calculate_offsets(table->ndim, table->shape, strides);
	for(i = 0; i < table->ndim; i++) {
		if(table->strides[i] != strides[i]) {
			NDTable_set_error_message("The offset[%d] of '%s' in '%s' must be %d but was %d", i, "table->datasetname", "table->filename", strides[i], table->strides[i]);
			return -1;
		}
	}

	// check the breakpoints
	for(i = 0; i < table->ndim; i++) {

		// make sure a scale is set
		if(table->breakpoints[i] == NULL) {
			NDTable_set_error_message("Scale for dimension %d of '%s' in '%s' is not set", i, "table->datasetname", "table->filename");
			return -1;
		}

		// check strict monotonicity
		v = table->breakpoints[i][0];
		for(j = 1; j < table->shape[i]; j++) {
			if(v >= table->breakpoints[i][j]) {
				NDTable_set_error_message("Scale for dimension %d of '%s' in '%s' is not strictly monotonic increasing at index %d", i, "table->datasetname", "table->filename", j);
				return -1;
			}
			v = table->breakpoints[i][j];
		}

		if(table->strides[i] != strides[i]) {
			NDTable_set_error_message("The offset[%d] of '%s' in '%s' must be %d but was %d", i, "table->datasetname", "table->filename", strides[i], table->strides[i]);
			return -1;
		}
	}

	// check the data for non-finite values
	for(i = 0; i < table->size; i++) {
		if(!ISFINITE(table->data[i])) {
			NDTable_set_error_message("The data value at index %d of '%s' in '%s' is not finite", i, "table->datasetname", "table->filename");
			return -1;
		}
	}

	return 0;
}

int NDTable_calculate_size(int ndim, const int shape[]) {
	int i, size = 1;

	for(i = 0; i < ndim; i++) {
		size = size * shape[i];
	}

	return size;
}

void NDTable_ind2sub(const int index, const NDTable_h table, int *subs) {
	int i, n = index; // number of remaining elements

	for(i = 0; i < table->ndim; i++) {
		subs[i] = n / table->strides[i];
		n -= subs[i] * table->strides[i];
	}
}

void NDTable_sub2ind(const int *subs, const NDTable_h table, int *index) {
	int i, k = 1;

	(*index) = 0;

	for(i = table->ndim-1; i >= 0; i--) {
		(*index) += subs[i] * k;
		k *= table->shape[i]; // TODO use pre-calculated offsets
	}
}

double NDTable_get_value_subs(const NDTable_h table, const int subs[]) {
	int index;
	NDTable_sub2ind(subs, table, &index);
	return table->data[index];
}

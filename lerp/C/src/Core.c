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

const char * NDTable_get_error_message() {
	return error_message;
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

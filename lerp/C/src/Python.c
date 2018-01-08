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

/*
Changes since fork from ScientificDataFormat NDTable

2017-12:
-------
	- remove data copy
	- naming as in numpy
	- derivative and interpolation
	- Pointer to Structure NDTable_h generation in Python
*/

#ifdef _WIN32
#define PYTHON_API __declspec(dllexport)
#else
#define PYTHON_API
#endif

#include <NDTable.h>
#include <stdio.h>

void free(void *ptr);

PYTHON_API int evaluate_derivative(
			NDTable_h mesh,
			const double **params,
			const double **delta_params,
			int ndimsparams,
			NDTable_InterpMethod_t interp_method,
			NDTable_ExtrapMethod_t extrap_method,
			int resultSize,
			double *result) {
	int i, j;
	double params_[32];
	double delta_params_[32];

	for(i = 0; i < resultSize; i++) {
		for(j = 0; j < ndimsparams; j++) {
			params_[j] = params[j][i];
			delta_params_[j] = delta_params[j][i];
		}

		if(NDT_eval_derivative(mesh, ndimsparams, params_, delta_params_, interp_method, extrap_method, &result[i]) != NDTABLE_INTERPSTATUS_OK) {
			return -1;
		}
	}

	return 0;

}

PYTHON_API int evaluate_interpolation(
			NDTable_h mesh,
			const double **params,
			int ndimsparams,
			NDTable_InterpMethod_t interp_method,
			NDTable_ExtrapMethod_t extrap_method,
			int resultSize,
			double *result) {
	int i, j;
	double params_[32];

	for(i = 0; i < resultSize; i++) {
		for(j = 0; j < ndimsparams; j++) {
			params_[j] = params[j][i];
		}
		if(NDT_eval(mesh, ndimsparams, params_, interp_method, extrap_method, &result[i]) != NDTABLE_INTERPSTATUS_OK) {
			return -1;
		}
	}

	return 0;
}

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


PYTHON_API NDTable_h make_table(
			double *data,
			const int ndim,
			double **breakpoints,
			const int *shape
			) {
	int i;

	NDTable_h table = NDTable_alloc_table();
	table->ndim = ndim;

	for(i = 0; i < table->ndim; i++) {
		table->shape[i] = shape[i];
		table->breakpoints[i] = breakpoints[i];
	}
	table->data = data;
	table->size = NDTable_calculate_size(table->ndim, table->shape);
	//table->itemsize = sizeof(table->data[0]);

	return table;
}


PYTHON_API int evaluate_derivative(
			double *data,
			int ndim,
			double **breakpoints,
			const int *shape,
			const double **params,
			int ndimsparams,
			NDTable_InterpMethod_t interp_method, // interpolation method
			NDTable_ExtrapMethod_t extrap_method, // extrapolation method
			int resultSize,
			double *result,
			double *delta_params) {

				/*
	int i, j;
	double params_[32];
	NDTable_h table = make_table(data, ndim, breakpoints, shape);

	for(i = 0; i < resultSize; i++) {
		for(j = 0; j < ndimsparams; j++) {
			params_[j] = params[j][i];
		}
	}

	return NDT_eval_derivative(table, ndimsparams, params_, delta_params, interp_method, extrap_method, result);
	*/
	return 0;
}


PYTHON_API int _derivative(
			NDTable_h mesh,
			const double params[],
			int ndimsparams,
			NDTable_InterpMethod_t interp_method, // interpolation method
			NDTable_ExtrapMethod_t extrap_method, // extrapolation method
			int resultSize,
			double *result,
			double delta_params[]) {

	return NDT_eval_derivative(mesh, ndimsparams, params, delta_params, interp_method, extrap_method, result);

}

PYTHON_API int evaluate_struct(
			NDTable_h mesh,
			const double **params,
			int ndimsparams,
			NDTable_InterpMethod_t interp_method, // interpolation method
			NDTable_ExtrapMethod_t extrap_method, // extrapolation method
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





/*
PYTHON_API int evaluate(

			double *data, // self.d concerted to np.float64
			const int ndim, // self.d.ndim
			double **breakpoints, //
			const int *shape,
			const double **params,
			int ndimsparams,
			NDTable_InterpMethod_t interp_method, // interpolation method
			NDTable_ExtrapMethod_t extrap_method, // extrapolation method
			int resultSize,
			double *result) {
	int i, j;
	double params_[32];

	NDTable_h table = make_table(data, ndim, breakpoints, shape);

	for(i = 0; i < resultSize; i++) {
		for(j = 0; j < ndimsparams; j++) {
			params_[j] = params[j][i];
		}

		if(NDT_eval(table, ndimsparams, params_, interp_method, extrap_method, &result[i]) != NDTABLE_INTERPSTATUS_OK) {
			NDTable_free_table(table);
			return -1;
		}
	}

	NDTable_free_table(table);
	return 0;
}
*/

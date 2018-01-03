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
	- link in Python.c removed -> need to be re-writen
*/


// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the PYTHON_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// PYTHON_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef _WIN32
#define PYTHON_API __declspec(dllexport)
#else
#define PYTHON_API
#endif

#include "NDTable.h"


PYTHON_API NDTable_h create_table(int ndims, const int *dims, const double *data, const double **scales);

PYTHON_API void close_table(NDTable_h table);

PYTHON_API int evaluate(
	NDTable_h table,
	int ndims,
	const double **params,
	NDTable_InterpMethod_t interp_method,
	NDTable_ExtrapMethod_t extrap_method,
	int nvalues,
	double *values);

PYTHON_API int evaluate_derivative(
	NDTable_h table,
	int nparams,
	const double params[],
	const double delta_params[],
	NDTable_InterpMethod_t interp_method,
	NDTable_ExtrapMethod_t extrap_method,
	double *value);

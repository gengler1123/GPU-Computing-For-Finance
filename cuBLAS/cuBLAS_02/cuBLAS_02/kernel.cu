
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#include "cublas_v2.h"

const int n = 6;

int main()
{
	cublasHandle_t handle;
	int j;
	float *x;
	x = new float[n];

	for (j = 0; j < n; j++)
	{
		x[j] = float(j);
	}

	float *d_x;
	cudaMalloc((void**)&d_x, n*sizeof(float));

	cublasCreate(&handle);
	cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
	float result;

	cublasSasum(handle, n, d_x, 1, &result);
	
	std::cout << "The Sum of the Absolute Value of the Elements Is: " << result << std::endl;

	cudaFree(d_x);
	cublasDestroy(handle);
	delete[] x;

    return 0;
}

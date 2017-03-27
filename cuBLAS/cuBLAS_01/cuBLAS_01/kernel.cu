
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
const int n = 6;

int main()
{
	cublasStatus_t stat;
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
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
	int result;

	stat = cublasIsamax(handle, n, d_x, 1, &result);

	std::cout << "The Largest Value is : " << x[result-1] << std::endl;

	cudaFree(d_x); 

	cublasDestroy(handle);

	delete[] x;

    return 0;
}

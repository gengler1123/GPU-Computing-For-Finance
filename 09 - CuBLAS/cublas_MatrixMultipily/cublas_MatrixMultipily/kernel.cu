
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <iostream>

#define IDX2C(i,j,ld) (((j)*(ld)) + (i))

const int m = 100;
const int n = 125;



int main()
{

	std::cout << "Performing y = alpha * A * x + beta*y " << std::endl;
	std::cout << "Where alpha, beta are scalars" << std::endl;
	std::cout << "x,y are vectors, and A is a matrix" << std::endl;
	std::cout << std::endl;

	cublasHandle_t handle;
	
	float *a = new float[m*n];
	float *x = new float[n];
	float *y = new float[m];

	int ind = 11;
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < m; i++)
		{
			a[IDX2C(i, j, m)] = float(ind++);
		}
	}


	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			std::cout << a[IDX2C(i, j, m)] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	for (int i = 0; i < n; i++)
	{
		x[i] = 1.0f;
	}
	for (int j = 0; j < m; j++)
	{
		y[j] = 0.0f;
	}


	float *d_a, *d_x, *d_y;

	cudaMalloc((void**)&d_a, m*n*sizeof(float));
	cudaMalloc((void**)&d_x, n*sizeof(float));
	cudaMalloc((void**)&d_y, m*sizeof(float));

	cublasCreate(&handle);
	cublasSetMatrix(m, n, sizeof(float), a, m, d_a, m);

	cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
	cublasSetVector(m, sizeof(float), y, 1, d_y, 1);

	float alpha = 1.0f, beta = 1.0f;

	cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_a, m, d_x, 1, &beta, d_y, 1);

	cublasGetVector(m, sizeof(float), d_y, 1, y, 1);

	for (int i = 0; i < m; i++)
	{
		std::cout << y[i] << std::endl;
	}

	cudaFree(d_a); cudaFree(d_x); cudaFree(d_y);
	cublasDestroy(handle);
	delete[] a; delete[] x; delete[] y;

    return 0;
}

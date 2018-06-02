#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>

int main()
{
	cublasHandle_t handle;

	cublasCreate(&handle);

	cublasDestroy(handle);

    return 0;
}


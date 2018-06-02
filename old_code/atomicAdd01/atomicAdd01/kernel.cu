
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <iostream>

#define numThreads 128
const int N = pow(2, 5);


#include <iostream>

__global__  void testAdd(float *a)
{
	for (int i = 0; i < 100; i++)
	{
		atomicAdd(&a[i], 1.0f);
	}
}
void cuTestAtomicAdd(float *a)
{
	testAdd << <1, 10 >> >(a);
}

int main(){

	float *d_data, *h_data;

	h_data = (float *)malloc(100 * sizeof(float));

	cudaMalloc((void **)&d_data, 100 * sizeof(float));

	cudaMemset(d_data, 0, 100 * sizeof(float));

	cuTestAtomicAdd(d_data);

	cudaMemcpy(h_data, d_data, 100 * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 100; i++)
		if (h_data[i] != 10.0f) { printf("mismatch at %d, was %f, should be %f\n", i, h_data[i], 10.0f); return 1; }

	printf("Success\n");
	return 0;
}


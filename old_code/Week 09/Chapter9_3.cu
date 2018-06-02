
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "C:/Users/FSCLAPTOP06/Desktop/laptop/Parisa_Desktop/PhD/Six Semester/FE 522 - Cuda/cuda_by_example/common/book.h"
#include <stdio.h>

 
#define size (100 *1024*1024) 
// Everything is exactly the same as the previous file(Chapter9-2)
// the core problem about previous file was that thousands of threads were competing for access to a relatively small number of memory addresses. 
// To address this issue, we will split our histogram computation into two phases
// Phase 1: 
//    Each parallel block will compute a separate histogram of the data that its constituent threads examine.
//    Since each block does this independently, we can compute these histograms in shared memory
//    We still need atomic operations because multiple threads want to examine the same data elements
//    The fact that only 256 threads will now be competing for 256 addresses will reduce contention
//    We need to create a shared memory buffer "temp[]" instead of the global memory buffer "histo[]"
//    Set the shared memory buffer to zero (zeroing the histogram)
//    So this phase involves with allocation and zeroing a shared memory buffer to hold each blocks histogram
//    we need syncthreads() to ensure every threads write has completed before progressing
//    then everything with be the same as before for GPU histogram computation except that we use temp instead of histo
// Phase 2:
//    we need to merge each blocks histogram to the global buffer histo
//    So each bin of the final histogram is the sum of the corresponding bin in all of the separated histograms
//    This addition needs to be done atomically.
//    Since we have decided to use 256 threads and have 256 bins, each thread atomically adds a single bin to the final histogram 
// In this form of computation, adding the shared memory reduces the running time  
__global__ void histo_kernel(unsigned char *buffer1, long size1, unsigned int *histo1){
	
	// Phase 1 ------------------------------------------------------------
	__shared__ unsigned int temp[256];
	temp[threadIdx.x] = 0;
	__syncthreads();
 
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (i < size1){
		atomicAdd(&(temp[buffer1[i]]),1);
		i += stride;
	}
	__syncthreads();
	//---------------------------------------------------------------------

	// Phase 2 ------------------------------------------------------------
	atomicAdd(&(histo1[threadIdx.x]), temp[threadIdx.x]);
	//---------------------------------------------------------------------
}

int main(void){
	// Step 1:
	cudaEvent_t start,stop;
	HANDLE_ERROR(cudaEventCreate(&start)); // dont forget & in cudaEventCreate
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start,0));
	unsigned char *buffer = (unsigned char*)big_random_block(size); // buffer is like an array contains random characters
	
	// Step 2:---------------------------------------------
	unsigned char *dev_buffer;
	unsigned int *dev_histo;
	HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, size));
	HANDLE_ERROR(cudaMalloc((void**)&dev_histo, 256 * sizeof(long)));
	HANDLE_ERROR(cudaMemset(dev_histo, 0, 256 * sizeof(int)));
	//-----------------------------------------------------

	// Step 3:---------------------------------------------
	HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, size, cudaMemcpyHostToDevice));

	//-----------------------------------------------------

	// Step 8----------------------------------------------
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
	int blocks = prop.multiProcessorCount;
	histo_kernel<<<blocks *2 , 256>>>(dev_buffer,size,dev_histo);
	//-----------------------------------------------------
	

	// Step 4:---------------------------------------------
	unsigned int histo[256];
	HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));
	//-----------------------------------------------------

	// Step 5:---------------------------------------------
	HANDLE_ERROR(cudaEventRecord(stop,0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));
	printf("Time to generate: %3.1f ms\n", elapsedTime);
	//-----------------------------------------------------
	
	// Step 9: calculate sum of the histo values-----------
	long histoCount = 0;
	for (int i = 0; i <256; i++){
		histoCount += histo[i];
	}
	printf("Histogram Sum : %ld\n", histoCount);
	//-----------------------------------------------------

	// Step 6----------------------------------------------
	for (int i = 0; i < size; i ++){
		histo[buffer[i]]--;
	}
	for (int i = 0; i < 256; i++){
		if (histo[i] != 0)
			printf("Failure at %d!\n", i);
	}
	//-----------------------------------------------------

	// Step 7----------------------------------------------
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	cudaFree(dev_histo);
	cudaFree(dev_buffer);
	//-----------------------------------------------------
	

	
	free(buffer);
	return 0;
}

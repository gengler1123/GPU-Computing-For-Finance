
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "C:/Users/FSCLAPTOP06/Desktop/laptop/Parisa_Desktop/PhD/Six Semester/FE 522 - Cuda/cuda_by_example/common/book.h"
#include <stdio.h>

 
#define size (100 *1024*1024) 
// Everything is exactly the same as the previous file(Chapter9-1)
// 1. Lets calculate the time
//    In order to do this, we need to create events of type cudaEvent_t
//    whenever you create these events, dont forget to use HANDLE_ERROR() function
// 2. Now that we set up our input data and events, we need to look to GPU memory.
//    we need to allocate memory in GPU for our random input data
//    We need to allocate memory in GPU for our output histogram
//    We need to initialize our histogram by 0 in GPU
//    Thats why we have to create pointers that points to the device
// 3. Then we have to copy our input which is buffer to the GPU. 
// 4. We need to copy histogram from GPU to Host.
// 5. Now that we are done with computation, we need to stop the timer
//    We can display the elapsed time
// 6. Now lets verify that we got the correct result
//    Lets compare the result with histo obtained in CPU
//    lets compute the histogram in CPU in reverse 
//    In reverse means, use the histo obtained in GPU and decreament each bin and see if we get to zero for all bins
// 7. Clean up everything:
//    clean up allocated cuda events
//    clean up GPU memory
//    clean up host memory
// 8. Now you can call the kernel function
//    before calling the kernel, you have to decide how to allocate threads and blocks
//    the optimal performance is achieved when the number of blocks is twice the number of multiprocessors our GPU contains.
//    Use a device property to obtain number of multiprocessors in GPU
// 9. create your kernel function
//    the first important thing: parameters of kernel function
//    how to allocate the ids for thread and block
//    Each thread will start with an offset between 0 and the number of threads minus 1
//    Then, it will stride by the total number of threads that have been launched
//    Once each thread knows its starting offset i and the stride it should use, the code walks through the input array incrementing the corresponding histogram bins
//    atomicAdd(address,y) generates an atomic sequence of operations that read the value at "address" , adds "y" to the value and stores the result back to the memory address "address".
//    No other thread can read or write the value at this address while we perform these operations
//    In this code, the address is the location of the histogram bin that corresponds to the current byte.
//    The current byte is buffer1[i]
//    The corresponding histogram bin is histo1[buffer1[i]]
__global__ void histo_kernel(unsigned char *buffer1, long size1, unsigned int *histo1){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (i < size1){
		atomicAdd(&(histo1[buffer1[i]]),1);
		i += stride;
	}
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
	free(buffer);
	//-----------------------------------------------------
	
	return 0;
}

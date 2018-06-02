
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


int main()
{
	int numDev = 0;
	cudaGetDeviceCount(&numDev);
	std::cout << "There are " << numDev << " GPU's in this system." << std::endl;

	cudaSetDevice(0);

	cudaDeviceEnablePeerAccess(1, 0);

	cudaSetDevice(1);

	cudaDeviceEnablePeerAccess(0, 0);

	/*
	
	Run Code Using Both
	
	*/
	cudaSetDevice(0);
	cudaDeviceDisablePeerAccess(1);
	
	cudaSetDevice(1);
	cudaDeviceDisablePeerAccess(0);



    return 0;
}

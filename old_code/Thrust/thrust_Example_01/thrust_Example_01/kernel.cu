#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>

int main(void)
{
	thrust::host_vector<int> h_vec(1000000);
	std::generate(h_vec.begin(), h_vec.end(), rand);

	// transfer data to the device
	thrust::device_vector<int> d_vec = h_vec;


	thrust::sort(d_vec.begin(), d_vec.end());

	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	for (int i = 0; i < 10; i++)
	{
		std::cout << h_vec[i] << std::endl;
	}

	for (int i = 0; i < 10; i++)
	{
		std::cout << d_vec[i] << std::endl;
	}

	return 0;
}
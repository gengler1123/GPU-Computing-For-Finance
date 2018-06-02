#include <thrust/scan.h>

#include <thrust/device_vector.h>
#include <thrust/count.h>

#include <thrust/functional.h>

#include <iostream>
#include <iomanip>

struct not_zero{
	__host__ __device__ bool operator()(const float a)
	{
		return a != 0;
	}
};


int main()
{
	thrust::device_vector<int> V(10, 0);
	V[0] = 1;
	V[3] = 5;
	V[5] = 10;
	V[7] = 15;
	V[9] = 25;

	int result = thrust::count(V.begin(), V.end(), 0);
	std::cout << "Zero Occured " << result << " Times." << std::endl;

	result = thrust::count_if(V.begin(), V.end(), not_zero());

	std::cout << "There were " << result << " non-zero values." << std::endl;


	thrust::device_vector<int> E(10);
	thrust::device_vector<int> I(10);

	thrust::inclusive_scan(V.begin(), V.end(), I.begin());
	thrust::exclusive_scan(V.begin(), V.end(), E.begin());

	
	std::cout << "V     I     E" << std::endl;
	for (int i = 0; i < V.size(); i++)
	{
		std::cout << V[i] << "  ,  " << I[i] << "  ,  " << E[i] << std::endl;
	}

}

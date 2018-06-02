#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include <iostream>




int main()
{
	thrust::device_vector<int> V;
	V.push_back(10);
	V.push_back(4);
	V.push_back(7);
	V.push_back(20);

	std::cout << "Pre-Sorted" << std::endl;
	for (int i = 0; i < V.size(); i++)
	{
		std::cout << V[i] << std::endl;
	}

	thrust::sort(V.begin(), V.end());

	std::cout << "\nSorted" << std::endl;
	for (int i = 0; i < V.size(); i++)
	{
		std::cout << V[i] << std::endl;
	}

	thrust::sort(V.begin(), V.end(), thrust::greater<int>());

	std::cout << "\nGreater Sort" << std::endl;
	for (int i = 0; i < V.size(); i++)
	{
		std::cout << V[i] << std::endl;
	}
}
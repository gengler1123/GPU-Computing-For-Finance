#include <iostream>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <thrust/inner_product.h>

int main()
{

	thrust::device_vector<int> V(5, 0);

	V[0] = 1;

	V[2] = 5;

	V[4] = 9;


	int sum = thrust::reduce(
		V.begin(), 
		V.end(), 
		(int)0, 
		thrust::plus<int>());

	std::cout << "The Sum Is " << sum << "." << std::endl;

	thrust::device_vector<int> W(5, 1);

	int inner = thrust::inner_product(V.begin(), V.end(), W.begin(), 0.0);

	std::cout << "The inner product of V with W is " << inner << std::endl;




	thrust::device_vector<int>::iterator iter = thrust::max_element(
		V.begin(), 
		V.end());

	std::cout << "The maximum value is " << *iter << ", at " << iter - V.begin() << "." << std::endl;

}
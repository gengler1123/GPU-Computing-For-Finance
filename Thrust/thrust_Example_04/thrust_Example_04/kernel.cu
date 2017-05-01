#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include <iostream>


struct saxpy_func
{
	const float a;

	saxpy_func(float a_) :a(a_){}

	__host__ __device__ float operator()(const float &x, const float&y)
		const{
		return a * x + y;
	}
};


void saxpy_fast(float A, thrust::device_vector<float> &X, thrust::device_vector<float> &Y)
{
	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_func(A));
}


int main()
{
	thrust::device_vector<float> X;
	thrust::device_vector<float> Y;

	for (int i = 0; i < 100; i++)
	{
		X.push_back(1);
		Y.push_back(i);
	}

	saxpy_fast(3, X, Y);

	for (int i = 0; i < 100; i++)
	{
		std::cout << Y[i] << std::endl;
	}

}
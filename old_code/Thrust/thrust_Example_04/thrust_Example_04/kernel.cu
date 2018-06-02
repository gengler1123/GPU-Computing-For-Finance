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


void saxpy_fast(
	float A, 
	thrust::device_vector<float> &X, 
	thrust::device_vector<float> &Y)
{
	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_func(A));
}


int main()
{
	thrust::host_vector<float> hX;
	thrust::host_vector<float> hY;

	for (int i = 0; i < 100; i++)
	{
		hX.push_back(1);
		hY.push_back(i);
	}

	thrust::device_vector<float> X = hX;
	thrust::device_vector<float> Y = hY;

	saxpy_fast(3, X, Y);

	hY = Y;

	for (int i = 0; i < 100; i++)
	{
		std::cout << hY[i] << std::endl;
	}

}
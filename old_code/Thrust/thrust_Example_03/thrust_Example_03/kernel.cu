#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include<thrust/transform.h>
#include<thrust/functional.h>


#include <iostream>


struct saxpi{
	float k1;
	float k2;
	saxpi(float k1_, float k2_) :k1(k1_), k2(k2_)
	{}
	__host__ __device__ float operator()(float &x)const{
		return x*k1 + k2;
	}
};


int main()
{
	thrust::host_vector<int> h_vec;
	h_vec.push_back(10);

	thrust::device_vector<int> d_vec;
	d_vec = h_vec;
	d_vec.push_back(1000);

	thrust::host_vector<int> h_vec1 = d_vec;

	std::cout << h_vec1[0] << "," << h_vec1[1] << std::endl;







	thrust::host_vector<float> V;
	for (int i = 0; i < 20; i++)
	{
		V.push_back(float(i));
	}

	saxpi f(2, 5);

	thrust::device_vector<float> D = V;

	thrust::device_vector<float> vH(V.size());

	thrust::transform(D.begin(), D.end(), vH.begin(), f);

	for (int i = 0; i < vH.size(); i++)
	{
		std::cout << vH[i] << std::endl;
	}

}
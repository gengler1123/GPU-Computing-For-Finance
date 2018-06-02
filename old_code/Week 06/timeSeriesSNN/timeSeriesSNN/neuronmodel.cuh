#ifndef __NEURONMODEL_CUH__
#define __NEURONMODEL_CUH__

#include <iostream>


class NeuronModel
{
private:
	/* Member Variables */
	float v, u, dt, hv, hu, I, C;
	float k, a, b, c, d;
	float v_r, v_t;
	float peak;
	int T;
	float x, y;

	/* Member Functions */
	__device__ __host__ float dv()
	{
		return (k*(v - v_r)*(v - v_t) - u + I)*C;
	};

	__device__ __host__ float du()
	{
		return a*(b*(v - v_r) - u);
	};

	__device__ __host__ bool checkPeak()
	{
		if (v >= peak)
		{
			return true;
		}
		return false;
	};

	__device__ __host__ void setFired()
	{
		fired = true;
		v = c;
		u += d;
	};
	friend class SNNetwork;
public:
	/* Member Variables */
	bool fired = false;

	/* Constructor */
	__device__ __host__ NeuronModel(
		float dt_ = 0.25)
	{
		v = -55.0;
		u = 0.0;
		dt = dt_;
		T = int(1 / dt);
		I = 0;
	};

	/* Member Functions */
	__device__ __host__ void timestep()
	{
		fired = false;
		//std::cout << "I = " << I << std::endl;
		for (int t = 0; t < T; t++)
		{
			hv = dv();
			//std::cout << "hv = " << hv << std::endl;
			hu = du();
			//std::cout << "hu = " << hu << std::endl;
			v += dt * hv;
			u += dt * hu;
			//std::cout << v << "," << u << std::endl;
			//std::cout << std::endl;
			if (checkPeak())
			{
				setFired();
				break;
			}
		}
	}

	__device__ __host__ void resetInput()
	{
		I = 0;
	};
	__device__ __host__ void updateInput(float w_)
	{
		I += w_;
	};

	/* Initialization Functions */
	__device__ __host__ void initializeExcitatory()
	{
		k = 0.7;
		v_r = -60;
		v_t = -40;
		a = 0.03;
		b = -2.0;
		c = -50.0;
		d = 100.0;
		peak = 35;
		C = 1 / 100.0;
	};

	__device__ __host__ void initializeInhibitory()
	{
		k = 1.5;
		v_r = -60.0;
		v_t = -40.0;
		a = 0.03;
		b = 1.0;
		c = -40.0;
		d = 150.0;
		peak = 25.0;
		C = 1 / (50.0);
	};

	/* Spatial Functions */
	__device__ __host__ void setCoordinates(float x_, float y_)
	{
		x = x_;
		y = y_;
	}

	__device__ __host__ float dist(NeuronModel NM)
	{
		return sqrt((x - NM.x)*(x - NM.x) + (y - NM.y)*(y - NM.y));
	}

	__device__ __host__ void printState()
	{
		std::cout << v << "," << u << std::endl;
	}

	__device__ __host__ void printParameters()
	{
		std::cout << "k : " << k << std::endl;
		std::cout << "vr: " << v_r << std::endl;
		std::cout << "vt: " << v_t << std::endl;
		std::cout << "a : " << a << std::endl;
		std::cout << "b : " << b << std::endl;
		std::cout << "c : " << c << std::endl;
		std::cout << "d : " << d << std::endl;
		std::cout << "fired : " << fired << std::endl;
	}

	__host__ void printCoor()
	{
		std::cout << x << "," << y << std::endl;
	}
};



#endif
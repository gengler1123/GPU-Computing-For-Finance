
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <random>
#include <ctime>
#include <string>

/* CUDA Includes */
#include "neuronmodel.cuh"
#include "kernels.cuh"
#include "edges.cuh"
/* HOST Includes */
#include "inputfuncs.cuh"

#include "snnetwork.cuh"

int main(int argc, char **argv)
{
	int numNeurons, runTime, maxDelay, numThreads, transientTime;
	float prop_excit, p_e2e, p_e2i, p_i2e, p_i2i, w_e_max, w_i_max, tau_e, tau_i, x_max, y_max, dt;
	std::string PATH;

	if (argc == 19)
	{

		std::cout << "Accepting Outside Parameters" << std::endl;

		numNeurons = std::stoi(argv[1]);
		runTime = std::stoi(argv[2]);
		maxDelay = std::stoi(argv[3]);
		numThreads = std::stoi(argv[4]);
		transientTime = std::stoi(argv[5]);
		PATH = argv[6];
		prop_excit = std::stof(argv[7]);
		p_e2e = std::stof(argv[8]);
		p_e2i = std::stof(argv[9]);
		p_i2e = std::stof(argv[10]);
		p_i2i = std::stof(argv[11]);
		w_e_max = std::stof(argv[12]);
		w_i_max = std::stof(argv[13]);
		tau_e = std::stof(argv[14]);
		tau_i = std::stof(argv[15]);
		x_max = std::stof(argv[16]);
		y_max = std::stof(argv[17]);
		dt = std::stof(argv[18]);

		std::cout << "INPUTED PARAMETERS:\n" << std::endl;
		std::cout << "Number of Neurons: " << numNeurons << std::endl;
		std::cout << "Running Time: " << runTime << std::endl;
		std::cout << "Maximum Delay: " << maxDelay << std::endl;
		std::cout << "Number of CUDA Threads: " << numThreads << std::endl;
		std::cout << "Transient Running Time: " << transientTime << std::endl;
		std::cout << "Results PATH: " << PATH << std::endl;
		std::cout << "Proportion Excitatory: " << prop_excit << std::endl;
		std::cout << "Prob. Excit -> Excit: " << p_e2e << std::endl;
		std::cout << "Prob. Excit -> Inhib: " << p_e2i << std::endl;
		std::cout << "Prob. Inhib -> Excit: " << p_i2e << std::endl;
		std::cout << "Prob. Inhib -> Inhib: " << p_i2i << std::endl;
		std::cout << "Excitatory Maximum Weight: " << w_e_max << std::endl;
		std::cout << "Inhibitory Maximum Weight: " << w_i_max << std::endl;
		std::cout << "Spatial Constant Excitatory: " << tau_e << std::endl;
		std::cout << "Spatial Constant Inhibitory: " << tau_i << std::endl;
		std::cout << "Maximum X Value: " << x_max << std::endl;
		std::cout << "Maximum Y Value: " << y_max << std::endl;
		std::cout << "Inner Neuron Timestep Value: " << dt << std::endl;
		std::cout << std::endl;

	}



	SNNetwork Network(
		numNeurons,
		runTime, maxDelay, numThreads, transientTime,
		PATH,
		prop_excit,
		p_e2e, p_e2i, p_i2e, p_i2i,
		w_e_max, w_i_max,
		tau_e, tau_i,
		time(NULL),
		x_max, y_max,
		dt,
		func1);

	Network.RunSimulation();
	Network.WriteNetworkStructure();

    return 0;
}

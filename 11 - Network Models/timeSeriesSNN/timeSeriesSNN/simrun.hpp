#ifndef __SIMRUN_HPP__
#define __SIMRUN_HPP__

#include "kernels.cuh"
#include "neuronmodel.cuh"
#include "edges.cuh"
#include <vector>

void transient(
	int runTime, int numNeurons,
	int numEdges, int numThreads,
	int maxDelay,
	NeuronModel *d_nm,
	Edge *d_ed,
	bool *d_cf,
	float *d_di,
	bool **Firings)
{
	std::vector<float> inputs(numNeurons);
	float *d_inputs;
	cudaMalloc((void**)&d_inputs, numNeurons*sizeof(float));
	for (int i = 0; i < inputs.size(); i++)
	{
		inputs[i] = 0.0f;
	}
	cudaMemcpy(d_inputs, inputs.data(), numNeurons*sizeof(float), cudaMemcpyHostToDevice);

	for (int t = 0; t < runTime; t++)
	{
		DelayedInput << < (numNeurons + numThreads - 1) / numThreads, numThreads >> >(
			numNeurons, maxDelay, (t%maxDelay),
			d_nm,
			d_di, d_inputs);

		Timestep << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
			d_nm,
			numNeurons,
			d_cf);

		cudaMemcpy(Firings[t], d_cf, numNeurons*sizeof(bool), cudaMemcpyDeviceToHost);
	}

	cudaFree(d_inputs);
}


void runSim(
	int runTime, int numNeurons,
	int numEdges, int numThreads,
	int maxDelay,
	NeuronModel *d_nm,
	Edge *d_ed,
	bool *d_cf,
	float *d_di,
	bool **Firings,
	void (*inputFunc)(int, std::vector<float>&))
{
	std::vector<float> inputs(numNeurons);
	float *d_inputs;
	cudaMalloc((void**)&d_inputs, numNeurons*sizeof(float));

	for (int t = 0; t < runTime; t++)
	{
		inputFunc(t, inputs);
		cudaMemcpy(d_inputs, inputs.data(), numNeurons*sizeof(float), cudaMemcpyHostToDevice);

		DelayedInput <<< (numNeurons + numThreads - 1)/numThreads , numThreads >>>(
			numNeurons, maxDelay, (t%maxDelay),
			d_nm,
			d_di, d_inputs);

		Timestep <<<(numNeurons + numThreads - 1) / numThreads, numThreads >>>(
			d_nm,
			numNeurons,
			d_cf);

		UpdateInputs <<< (numEdges + numThreads - 1)/numThreads, numThreads >>>(
			numEdges, (t%maxDelay), maxDelay,
			d_ed,
			d_cf,
			d_di);

		cudaMemcpy(Firings[t], d_cf, numNeurons*sizeof(bool), cudaMemcpyDeviceToHost);
	}

	cudaFree(d_inputs);

};

#endif
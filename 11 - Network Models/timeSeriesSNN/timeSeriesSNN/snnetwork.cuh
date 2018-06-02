#ifndef __SNNETWORK_HPP__
#define __SNNETWORK_HPP__

#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>

#include "kernels.cuh"

class SNNetwork
{
private:
	/* Member Variables */
	int numNeurons, numExcit, numInhib;
	int runTime, numEdges, maxDelay, numThreads, transientTime;
	std::string PATH;
	float prop_excit;
	float p_E2E, p_E2I, p_I2I, p_I2E;
	float w_E_max, w_I_max;
	float tau_E, tau_I;

	float dt;

	float x_max, y_max, maxDist;

	void(*inputFunc)(int, std::vector<float>&);


	/* Pseudo-Random Number Generation */
	std::mt19937 *gen;


	/* Member Objects */
	std::vector<NeuronModel> Neurons;
	NeuronModel *h_nm;
	std::vector<Edge> Edges;
	std::vector<float> h_di;
	bool **Firings;


	/* CUDA Objects */
	NeuronModel *d_nm;
	Edge *d_ed;
	float *d_di;
	bool *d_cf;


	/*
	Member Functions
	*/

	/* Network Structure Generation */
	void InitializeNeurons()
	{
		float x, y;
		std::uniform_real_distribution<float> x_dist(0.0, x_max), y_dist(0.0, y_max);

		for (int n = 0; n < numExcit; n++)
		{
			NeuronModel N;
			N.initializeExcitatory();
			x = x_dist(*gen);
			y = y_dist(*gen);
			N.setCoordinates(x, y);
			Neurons.push_back(N);
		}
		for (int n = 0; n < numInhib; n++)
		{
			NeuronModel N;
			N.initializeInhibitory();
			x = x_dist(*gen);
			y = y_dist(*gen);
			N.setCoordinates(x, y);
			Neurons.push_back(N);
		}
	};


	void InitializeEdges()
	{
		float d;
		float p;
		float r;
		std::uniform_real_distribution<float> dist(0.0, 1.0);

		for (int n = 0; n < numNeurons; n++)
		{
			for (int m = 0; m < numNeurons; m++)
			{
				if (n != m)
				{
					d = Neurons[n].dist(Neurons[m]);
					r = dist(*gen);
					
					if (n < numExcit && m < numExcit)
					{
						/* p_E2E */
						p = p_E2E * exp(-d*tau_E);
						if (r < p)
						{
							Edge E;
							E.source = n;
							E.target = m;
							E.delay = (maxDelay-1) * (d / maxDist)+1;
							E.weight = w_E_max * dist(*gen);
							Edges.push_back(E);
						}
					}
					else if (n < numExcit)
					{
						/* p_E2I */
						p = p_E2I * exp(-d*tau_E);
						if (r < p)
						{
							Edge E;
							E.source = n;
							E.target = m;
							E.delay = (maxDelay - 1) * (d / maxDist) + 1;
							E.weight = w_E_max * dist(*gen);
							Edges.push_back(E);
						}
					}
					else if (m < numExcit)
					{
						/* p_I2E */
						p = p_I2E * exp(-d*tau_I);
						if (r < p)
						{
							Edge E;
							E.source = n;
							E.target = m;
							E.delay = (maxDelay - 1) * (d / maxDist) + 1;
							E.weight = -w_I_max * dist(*gen);
							Edges.push_back(E);
						}
					}
					else
					{
						/* p_I2I */
						p = p_I2I * exp(-d*tau_I);
						if (r < p)
						{
							Edge E;
							E.source = n;
							E.target = m;
							E.delay = (maxDelay - 1) * (d / maxDist) + 1;
							E.weight = -w_I_max * dist(*gen);
							Edges.push_back(E);
						}
					}
				}
			}
		}
		numEdges = Edges.size();
	};


	void AllocateOnDevice()
	{
		cudaMalloc((void**)&d_nm, numNeurons*sizeof(NeuronModel));
		cudaMalloc((void**)&d_ed, numEdges * sizeof(Edge));
		cudaMalloc((void**)&d_di, numNeurons*maxDelay*sizeof(float));
		cudaMalloc((void**)&d_cf, numNeurons*sizeof(bool));
	};


	void CopyToDevice()
	{
		cudaMemcpy(d_nm, Neurons.data(), numNeurons * sizeof(NeuronModel), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ed, Edges.data(), numEdges * sizeof(Edge), cudaMemcpyHostToDevice);
		cudaMemcpy(d_di, h_di.data(), numNeurons * maxDelay * sizeof(float), cudaMemcpyHostToDevice);
	};


	/* Network Runs */
	void Transient()
	{
		for (int t = 0; t < transientTime; t++)
		{
			Timestep <<<(numNeurons + numThreads - 1) / numThreads, numThreads >>>(
				d_nm,
				numNeurons,
				d_cf);
		}
	};


	void RunSim(
		void (*inputFunc)(int, std::vector<float>&))
	{
		std::vector<float> inputs(numNeurons);
		for (int i = 0; i < inputs.size(); i++)
		{
			inputs[i] = 0.0f;
		}
		float *d_inputs;
		cudaMalloc((void**)&d_inputs, numNeurons*sizeof(float));

		
		for (int t = 0; t < runTime; t++)
		{
			inputFunc(t, inputs);
			cudaMemcpy(d_inputs, inputs.data(), numNeurons*sizeof(float), cudaMemcpyHostToDevice);

			DelayedInput << < (numNeurons + numThreads - 1) / numThreads, numThreads >> >(
				numNeurons, maxDelay, (t%maxDelay),
				d_nm,
				d_di, d_inputs);

			Timestep << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
				d_nm,
				numNeurons,
				d_cf);

			UpdateInputs << < (numEdges + numThreads - 1) / numThreads, numThreads >> >(
				numEdges, (t%maxDelay), maxDelay,
				d_ed,
				d_cf,
				d_di);

			cudaMemcpy(Firings[t], d_cf, numNeurons*sizeof(bool), cudaMemcpyDeviceToHost);
		}
		cudaFree(d_inputs);
	};


	void WriteResults()
	{
		std::ofstream myFile(PATH);

		for (int t = 0; t < runTime; t++)
		{
			for (int n = 0; n < numNeurons; n++)
			{
				if (Firings[t][n])
				{
					myFile << t << "," << n << "\n";
				}
			}
		}

		myFile.close();
	};
	


public:
	/* Constructor */
	SNNetwork(
		int numNeurons_,
		int runTime_, int maxDelay_, int numThreads_, int transientTime_,
		std::string PATH_,
		float prop_excit_,
		float p_e2e_, float p_e2i_, float p_i2e_, float p_i2i_,
		float w_e_max_, float w_i_max_,
		float tau_e_, float tau_i_,
		unsigned int seed,
		float x_max_, float y_max_,
		float dt_,
		void(*inputFunc_)(int, std::vector<float>&)) :
		numNeurons(numNeurons_),
		runTime(runTime_),
		maxDelay(maxDelay_),
		numThreads(numThreads_),
		transientTime(transientTime_),
		PATH(PATH_),
		prop_excit(prop_excit_),
		p_E2E(p_e2e_),
		p_E2I(p_e2i_),
		p_I2E(p_i2e_),
		p_I2I(p_i2i_),
		w_E_max(w_e_max_), w_I_max(w_i_max_),
		tau_E(tau_e_), tau_I(tau_i_),
		x_max(x_max_), y_max(y_max_),
		dt(dt_),
		inputFunc(inputFunc_)
	{

		gen = new std::mt19937(seed);
		numExcit = prop_excit * numNeurons;
		numInhib = numNeurons - numExcit;
		Firings = new bool*[runTime];
		maxDist = sqrt(x_max * x_max + y_max*y_max);
		for (int t = 0; t < runTime; t++)
		{
			Firings[t] = new bool[numNeurons];
		}

		InitializeNeurons();
		InitializeEdges();
		AllocateOnDevice();
		CopyToDevice();

		h_nm = new NeuronModel[numNeurons];
		std::uniform_real_distribution<float> dist(0.0, 1.0);
		std::cout << dist(*gen) << std::endl;

		std::cout << "Seed      : " << seed << std::endl;
		std::cout << "Neurons   : " << numNeurons << std::endl;
		std::cout << "Num Excit : " << numExcit << std::endl;
		std::cout << "Num Inhib : " << numInhib << std::endl;
		std::cout << "Edges   : " << numEdges << std::endl;

	};

	/* Destructor */
	~SNNetwork()
	{
		std::cout << "Cleaning Up DEVICE Pointers" << std::endl;
		cudaFree(d_nm); cudaFree(d_di); cudaFree(d_ed); cudaFree(d_cf);
		cudaDeviceReset();

		std::cout << "Cleaning up HOST Pointers" << std::endl;
		delete gen;
		delete[] h_nm;
		for (int t = 0; t < runTime; t++)
		{
			delete[] Firings[t];
		}
		delete[] Firings;
	};


	/* Run Sim */
	void RunSimulation()
	{
		Transient();
		RunSim(inputFunc);
		WriteResults();
	}

	void printState()
	{
		cudaMemcpy(h_nm, d_nm, numNeurons*sizeof(NeuronModel), cudaMemcpyDeviceToHost);
		h_nm[0].printState();
		h_nm[0].printParameters();
	}


	void WriteNetworkStructure()
	{
		cudaMemcpy(h_nm, d_nm, numNeurons * sizeof(NeuronModel), cudaMemcpyDeviceToHost);
		std::ofstream myFile("netStructure.csv");
		myFile << "Source,Target,Weight,Delay\n";
		for (int i = 0; i < numEdges; i++)
		{
			myFile << Edges[i].source << "," << Edges[i].target << "," << Edges[i].weight << "," << Edges[i].delay << "\n";
		}
		myFile.close();

		std::ofstream nyFile("neurons.csv");
		nyFile << "ID,x,y\n";
		for (int i = 0; i < numNeurons; i++)
		{
			nyFile << i << ","<< h_nm[i].x << "," << h_nm[i].y << "\n";
		}
		nyFile.close();
	}




};




#endif
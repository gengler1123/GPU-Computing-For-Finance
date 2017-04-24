
#include "cuda_runtime.h"
#include "cusparse_v2.h"
#include "device_launch_parameters.h"


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cassert>

#define imin(a,b) (a<b?a:b)
#define imax(a,b) (a>b?a:b)

const int EFDN = 10;

void efdEuroCallPrice_cuda(float S, float sigma, float tau, float K, float r);

void efdForward_1d_BM_cuda(float S, float tau);

void efdForward_2d_BM_cuda(float S, float V, float tau);

void example2dArray();

void ifdEuroCallPrice_cuda(double S0, double sigma, double tau, double K, double r);

__global__ void EFD(
	int size,
	float *d_val_n,
	float *d_val_npo,
	float Pu, float Pm, float Pd,
	float x0, float x
	);

__global__ void EFD_1dBM(
	int size,
	float *d_val_n,
	float *d_val_npo,
	float Pu, float Pm, float Pd
	);

__global__ void EFD_2dBM(
	int width, int height, int pitch_n, int pitch_npo,
	float *d_val_n,
	float *d_val_npo,
	float alpha, float beta
	);

__global__ void modify_i_j(
	int width, int height, int pitch,
	float *d_array,
	int i, int j, float change_to
	);

__global__ void IFD_boundary(
	int size,
	double *d_Price,
	double lambda_U,
	double lambda_L
	);

int main(void) {

	//// Example 1: forward 1D BM
	//efdForward_1d_BM_cuda(100, 2);


	//// Example 2: backward 1d BS
	float S0 = 358.52, sigma = 0.230967, tau = 0.145205, K = 360.0, r = 0.06;
	efdEuroCallPrice_cuda(S0, sigma, tau, K, r);

	// Example 3: 2D array example
	//example2dArray();

	// Example 4: forward 2D BM
	//efdForward_2d_BM_cuda(100, 100, 2);

	// Example 5: backward 1D BS
	//float S0 = 358.52, sigma = 0.230967, tau = 0.145205, K = 360.0, r = 0.06;
	ifdEuroCallPrice_cuda(S0, sigma, tau, K, r);

}

void efdForward_2d_BM_cuda(float S, float V, float tau){
	//construct the 2D array
	float ST_max = S + 4 * sqrt(tau);
	const int width = 2 * EFDN + 1;
	float s = (ST_max - S) / (EFDN + 0.0);

	float VT_max = V + 4 * sqrt(tau);
	const int height = 2 * EFDN + 1;
	float v = (VT_max - V) / (EFDN + 0.0);

	float h_P0[width][height];

	//initial density:
	for (int i = 0; i < width; i++){
		for (int j = 0; j < height; j++){
			h_P0[i][j] = 0.0;
		}
	}
	h_P0[EFDN][EFDN] = 1.0;

	//time step
	int n = 100;
	float t = tau / n;

	//coefficients from the PDE:
	float alpha = t / 2.0 / s / s;
	float beta = t / 2.0 / v / v;

	//pass the 2D grid to device
	//what is pitch? http://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
	size_t h_pitch = width * sizeof(float); //host original array pitch in bytes, number of bytes in one row
	size_t d_pitch0, d_pitch1;// pitch for the device array 
	float *d_ptr0, *d_ptr1;
	cudaMallocPitch(&d_ptr0, &d_pitch0, width * sizeof(float), height);
	cudaMallocPitch(&d_ptr1, &d_pitch1, width * sizeof(float), height);
	cudaMemcpy2D(d_ptr0, d_pitch0, h_P0, h_pitch, width * sizeof(float), height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_ptr1, d_pitch1, h_P0, h_pitch, width * sizeof(float), height, cudaMemcpyHostToDevice);

	//calculate forward
	for (int i = 0; i < n; i++)
	{
		if (i % 2 == 0)
		{
			EFD_2dBM << < height, width >> >(
				width, height, d_pitch0, d_pitch1,
				d_ptr0,
				d_ptr1,
				alpha, beta
				);
		}
		else
		{
			EFD_2dBM << <height, width >> >(
				width, height, d_pitch1, d_pitch0,
				d_ptr1,
				d_ptr0,
				alpha, beta);
		}
	}

	//copy the result back to the host
	if ((n - 1) % 2 == 0){
		//cudaMemcpy(h_P0, d_P_1, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy2D(h_P0, h_pitch, d_ptr1, d_pitch1, width * sizeof(float), height, cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMemcpy2D(h_P0, h_pitch, d_ptr0, d_pitch0, width * sizeof(float), height, cudaMemcpyDeviceToHost);
	}

	//output the result
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			//std::cout << h_P0[i][j] << "\t";
			printf("%.2f\t", h_P0[i][j]);
		}
		std::cout << std::endl;
	}


}

__global__ void EFD_2dBM(
	int width, int height, int pitch_n, int pitch_npo,
	float *d_val_n,
	float *d_val_npo,
	float alpha, float beta
	){
	int idx = blockIdx.x;	//row
	int idy = threadIdx.x;	//column

	if ((idx < height) && (idy <width)){
		//d_val_npo[i] = Pu * d_val_n[i + 1] + Pm * d_val_n[i] + Pd * d_val_n[i - 1];
		d_val_npo[idx*(pitch_npo / sizeof(float)) + idy] = alpha*(d_val_n[(idx + 1)*(pitch_n / sizeof(float)) + idy]
			+ d_val_n[(idx - 1)*(pitch_n / sizeof(float)) + idy])
			+ beta*(d_val_n[idx*(pitch_n / sizeof(float)) + idy + 1]
			+ d_val_n[idx*(pitch_n / sizeof(float)) + idy - 1])
			+ (1.0 - 2.0*alpha - 2.0*beta)*d_val_n[idx*(pitch_n / sizeof(float)) + idy];

		//modify the ones on the top
		if (idx == 0){
			d_val_npo[idx*(pitch_npo / sizeof(float)) + idy] = d_val_npo[(idx + 1)*(pitch_npo / sizeof(float)) + idy];
		}
		//modify the ones on the bottom
		if (idx == (height - 1)){
			d_val_npo[idx*(pitch_npo / sizeof(float)) + idy] = d_val_npo[(idx - 1)*(pitch_npo / sizeof(float)) + idy];
		}
		//modify the ones on the left
		if (idy == 0){
			d_val_npo[idx*(pitch_npo / sizeof(float)) + idy] = d_val_npo[(idx - 1)*(pitch_npo / sizeof(float)) + idy + 1];
		}
		//modify the ones on the right
		if (idx == (width - 1)){
			d_val_npo[idx*(pitch_npo / sizeof(float)) + idy] = d_val_npo[(idx - 1)*(pitch_npo / sizeof(float)) + idy - 1];
		}
	}
}

void example2dArray(){
	std::cout << "Host main" << std::endl;

	// Host code
	const int width = 3;
	const int height = 3;
	float* devPtr;
	float a[width][height];

	//load and display input array
	std::cout << "a array: " << std::endl;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			a[i][j] = i + j;
			std::cout << a[i][j] << " ";

		}
		std::cout << std::endl;
	}
	std::cout << std::endl;


	//Allocating Device memory for 2D array using pitch
	size_t host_orig_pitch = width * sizeof(float); //host original array pitch in bytes
	size_t pitch;// pitch for the device array 

	cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);

	std::cout << "host_orig_pitch: " << host_orig_pitch << std::endl;
	std::cout << "sizeof(float): " << sizeof(float) << std::endl;
	std::cout << "width: " << width << std::endl;
	std::cout << "height: " << height << std::endl;
	std::cout << "pitch:  " << pitch << std::endl;
	std::cout << std::endl;

	cudaMemcpy2D(devPtr, pitch, a, host_orig_pitch, width * sizeof(float), height, cudaMemcpyHostToDevice);

	float b[width][height];
	//load b and display array
	std::cout << "b array: " << std::endl;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			b[i][j] = 0;
			std::cout << b[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;


	//MyKernel<<<100, 512>>>(devPtr, pitch, width, height);
	//cudaThreadSynchronize();
	modify_i_j << <height, width >> >(width, height, pitch, devPtr, 1, 1, 5); //one block for one row
	cudaThreadSynchronize();


	//cudaMemcpy2d(dst, dPitch,src ,sPitch, width, height, typeOfCopy )
	cudaMemcpy2D(b, host_orig_pitch, devPtr, pitch, width * sizeof(float), height, cudaMemcpyDeviceToHost);


	// should be filled in with the values of array a.
	std::cout << "returned array" << std::endl;
	for (int i = 0; i < width; i++){
		for (int j = 0; j < height; j++){
			std::cout << b[i][j] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
	system("pause");
}

__global__ void modify_i_j(
	int width, int height, int pitch,
	float *d_array,
	int i, int j, float change_to
	){
	//we want to change the [i,j]-th of the 2-dim array
	int idx = blockIdx.x; //row
	int idy = threadIdx.x; //column

	//we can do index by pointer:
	//if ((idx == i) && (idy == j)){
	//float* row = (float *)((char*)d_array + idx*pitch);
	//	row[idy] = change_to;
	//}

	//or, a more convenient way is to do index just use idx and idy
	if ((idx == i) && (idy == j))
	{
		d_array[idx*(pitch / sizeof(float)) + idy] = change_to;
	}

}


void efdForward_1d_BM_cuda(float S, float tau){

	//construct a proper grid for S
	float ST_max = S + 4 * sqrt(tau);
	int N = 2 * EFDN + 1;
	float s = (ST_max - S) / (EFDN + 0.0);
	float *h_P0;
	h_P0 = new float[N];

	//initialize the initial density
	for (int i = 0; i < N; i++){
		if (i == EFDN){
			h_P0[i] = 1.0;//a point mass at S
		}
		else{
			h_P0[i] = 0.0;
		}
	}

	//time step
	int n = 100;
	float t = tau / n;

	//coefficients from the PDE:
	float pu = t / 2.0 / s / s;
	float pd = t / 2.0 / s / s;
	float pm = 1.0 - t / s / s;

	//pass the grid to device:
	float *d_P_0, *d_P_1; // Device Pointers
	cudaMalloc((void**)&d_P_0, N * sizeof(float));
	cudaMalloc((void**)&d_P_1, N * sizeof(float));

	cudaMemcpy(d_P_0, h_P0, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_P_1, h_P0, N*sizeof(float), cudaMemcpyHostToDevice);

	// forward:
	for (int i = 0; i < n; i++)
	{
		if (i % 2 == 0)
		{
			EFD_1dBM << < 2, 40 >> >(
				N,
				d_P_0,
				d_P_1,
				pu, pm, pd
				);
		}
		else
		{
			EFD_1dBM << <2, 40 >> >(
				N,
				d_P_1,
				d_P_0, pu, pm, pd);
		}
	}

	if ((n - 1) % 2 == 0){
		cudaMemcpy(h_P0, d_P_1, N * sizeof(float), cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMemcpy(h_P0, d_P_0, N * sizeof(float), cudaMemcpyDeviceToHost);
	}

	std::cout << "the terminal density is:" << std::endl;
	for (int i = 0; i < N; i++){
		std::cout << S - EFDN*s + i*s << ": " << h_P0[i] << std::endl;
	}


}

__global__ void EFD_1dBM(
	int size,
	float *d_val_n,
	float *d_val_npo,
	float Pu, float Pm, float Pd
	){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		d_val_npo[i] = Pu * d_val_n[i + 1] + Pm * d_val_n[i] + Pd * d_val_n[i - 1];

		if (i == 0)
		{
			d_val_npo[i] = d_val_npo[1];
		}
		else if (i == size - 1)
		{
			d_val_npo[i] = d_val_npo[i - 1];
		}
	}
}

__global__ void EFD(
	int size,
	float *d_val_n,
	float *d_val_npo,
	float Pu, float Pm, float Pd,
	float x0, float x
	)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		d_val_npo[i] = Pu * d_val_n[i + 1] + Pm * d_val_n[i] + Pd * d_val_n[i - 1];

		if (i == 0)
		{
			d_val_npo[i] = d_val_npo[1];
		}

		else if (i == size - 1)
		{
			d_val_npo[i] = d_val_npo[i - 1]
				+ exp(x0 + x * (float(i / 2)))
				- exp(x0 + x * (float(i / 2 - 1)));
		}
	}
}

void efdEuroCallPrice_cuda(float S0, float sigma, float tau, float K, float r){

	//calculate all parameters in CPU
	float Tolerance = .001;
	float t = Tolerance / (1 + 3 * sigma*sigma);
	int n = tau / t;//n time intervals horizontally
	float x = sigma*sqrt(3 * t);
	int myN = 4 * sigma*sqrt(tau) / x;//	2N+1 possible values vertically
	std::cout << "myN=" << myN << std::endl;
	std::cout << "n=" << n << std::endl;


	float nu = r - .5*sigma*sigma;
	float disc = exp(-r*t);//discount factor

	float Pu = (sigma*sigma*t) / (2 * x*x) + (nu*t) / (2 * x);
	float Pd = (sigma*sigma*t) / (2 * x*x) - (nu*t) / (2 * x);
	float Pm = 1 - Pu - Pd;

	Pu = Pu*disc;
	Pm = Pm*disc;
	Pd = Pd*disc;


	float x0 = log(S0);

	int SIZEOFARR = 2 * EFDN + 1;

	// START NEW CODE

	float *h_Price; // Host Pointer
	float *d_Price_0, *d_Price_1; // Device Pointers

	/* Generate Terminal Conditions in h_Price */
	h_Price = new float[SIZEOFARR];
	for (int i = 0; i < SIZEOFARR; i++){
		float myx = x0 + x* (i + 1 - EFDN - 1);
		float myS = exp(myx);
		h_Price[i] = imax(0.0, myS - K);
		//std::cout << "h[" << i << "]=" << h_Price[i] << std::endl;
	}

	/* C */
	cudaMalloc((void**)&d_Price_0, SIZEOFARR * sizeof(float));
	cudaMalloc((void**)&d_Price_1, SIZEOFARR * sizeof(float));

	cudaMemcpy(d_Price_0, h_Price, SIZEOFARR*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Price_1, h_Price, SIZEOFARR*sizeof(float), cudaMemcpyHostToDevice);


	for (int i = n - 1; i >= 0; i--)
	{
		if (i % 2 == 0)
		{
			EFD << < 2, 40 >> >(
				SIZEOFARR,
				d_Price_0,
				d_Price_1,
				Pu, Pm, Pd, x0, x
				);
		}
		else
		{
			EFD << <2, 40 >> >(
				SIZEOFARR,
				d_Price_1,
				d_Price_0, Pu, Pm, Pd, x0, x);
		}
	}

	cudaMemcpy(h_Price, d_Price_1, SIZEOFARR * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "the efd price from device is " << h_Price[EFDN] << std::endl;

	delete[] h_Price;
}




void ifdEuroCallPrice_cuda(double S0, double sigma, double tau, double K, double r){

	double Tolerance = .001;
	double x = sqrt(Tolerance / 2.0);
	double t = Tolerance / 2.0;
	int n = tau / t;//n time intervals horizontally
	int N = 4 * sigma*sqrt(tau) / x;
	//int N = 100;

	//std::cout << "in ifd, N=" << N << std::endl;
	//	2N+1 possible values vertically

	//    cout<< "N="<<N<<endl<<"n="<<n<<endl;

	double nu; nu = r - .5*sigma*sigma;

	double alpha = -.5*t*(sigma*sigma / x / x + nu / x);
	double beta = 1.0 + t*sigma*sigma / x / x + r*t;
	double gamma = -.5*t*(sigma*sigma / x / x - nu / x);

	/*
	std::cout<<"alpha="<<alpha<<std::endl
	<<"beta="<<beta<<std::endl
	<<"gamma="<<gamma<<std::endl
	<<alpha+beta<<std::endl;
	*/

	double x0; x0 = log(S0);

	const int SIZEOFARR = 2 * EFDN + 1;


	// set up the 3 vectors of the tridiagonal matrix

	double *h_dl = (double*)malloc(SIZEOFARR*sizeof(double));
	double *h_d = (double*)malloc(SIZEOFARR*sizeof(double));
	double *h_du = (double*)malloc(SIZEOFARR*sizeof(double));

	for (int i = 0; i < SIZEOFARR; i++){
		if (i == 0){
			//first row
			h_dl[i] = 0.0;
			h_d[i] = 1.0;
			h_du[i] = -1.0;
		}
		else if (i == (SIZEOFARR - 1)){
			//last row
			h_dl[i] = 1.0;
			h_d[i] = -1.0;
			h_du[i] = 0.0;
		}
		else{
			//other rows
			h_dl[i] = alpha;
			h_d[i] = beta;
			h_du[i] = gamma;
		}
	}

	double *d_dl; cudaMalloc(&d_dl, SIZEOFARR*sizeof(double));
	double *d_d; cudaMalloc(&d_d, SIZEOFARR*sizeof(double));
	double *d_du; cudaMalloc(&d_du, SIZEOFARR*sizeof(double));

	cudaMemcpy(d_dl, h_dl, SIZEOFARR*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, h_d, SIZEOFARR*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_du, h_du, SIZEOFARR*sizeof(double), cudaMemcpyHostToDevice);



	/* Generate Terminal Conditions in h_Price */
	double *h_Price = (double *)malloc(SIZEOFARR*sizeof(double));
	//h_Price = new double[SIZEOFARR];

	for (int i = 0; i < SIZEOFARR; i++){
		
		double myx = x0 + x* ( EFDN - i );
		double myS = exp(myx);
		h_Price[i] = imax(0.0, myS - K);
	
	}

	double * d_Price; cudaMalloc(&d_Price, SIZEOFARR*sizeof(double));
	cudaMemcpy(d_Price, h_Price, SIZEOFARR*sizeof(double), cudaMemcpyHostToDevice);

	double lambda_U = exp(x0 + x*EFDN) - exp(x0 + x*(EFDN - 1));
	double lambda_L = 0.0;

	//initialize cuSPARSE
	cusparseHandle_t handle;	cusparseCreate(&handle);

	// sequential backward to the initial step, each step solves a tridiagonal system using cusparseSgtsv_nopivot
	for (int i = n-1 ; i >= 0; i--){
		//cusparseSgtsv_nopivot(cusparseHandle_t handle, int m, int n, const double *dl, const double *d, const double *du, double *B, int ldb)
		// a good example:
		// https://github.com/OrangeOwlSolutions/Linear-Algebra/blob/master/SolveTridiagonalLinearSystem.cu
		IFD_boundary << <2, 40 >> >(SIZEOFARR, d_Price, lambda_U, lambda_L);
		cusparseDgtsv(handle, SIZEOFARR, 1, d_dl, d_d, d_du, d_Price, SIZEOFARR);

	}

	// get the middle one as the reslting price

	cudaMemcpy(h_Price, d_Price, SIZEOFARR * sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << "the ifd price from device is " << h_Price[EFDN] << std::endl;

	/*for (int i = 0; i < SIZEOFARR; i++){
		std::cout << "h[" << i << "]=" << h_Price[i] << std::endl;
	}*/

	delete[] h_Price;
}


__global__ void IFD_boundary(
	int size,
	double *d_Price,
	double lambda_U,
	double lambda_L
	)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
	{
		if (i == 0)//top condition
		{
			d_Price[i] = lambda_U;
		}

		else if (i == size - 1) //bottom condition
		{
			d_Price[i] = 0.0;
		}
	}
}
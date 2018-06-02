
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "C:/Users/FSCLAPTOP06/Desktop/laptop/Parisa_Desktop/PhD/Six Semester/FE 522 - Cuda/cuda_by_example/common/book.h"
#include <stdio.h>

// 1. define the size of random numbers
// 2. In this main function, we need to create 100MB of random numbers 
//	  In order to create random numbers, use: "big_random_block()"
//    each byte has eight bits
//    each byte can be any of 256 different values from 0x00 to 0xFF(0 to 255)
// 3. This histogram needs to contain 256 bins in order to keep track of the number of times each value has been seen in the data
//    Thats why we need to create a 256-bin arrays
//    Initialize all the bin counts to zero
// 4. We need to tabulate the frequency with which each value appears in the data(data contained in buffer)
//    We want to increment bin z of our histogram
//    Lets count the number of times we have seen an occurrence of the value z.
// 5. If buffer[i] is the current value we are looking at, we want to increment the count we have in the bin numbered buffer[i]
//    but bin is located at histo[buffer[i]], so we need to increment this
//    So for each element in buffer, do this increament in histo
// 6. lets say this histogram is the input to the next step
//    lets say we want to verify that all the bins of our histogram sum to the expected value
//    The result will be the same and it will be equal to the "size" because it shows the total number of elements we have examined
//    Even if I change the size to 10 or 100 or 1000, it will just show the "size" as a result
// 7. we need to clean the buffer

// Step 1: size of random data: 1 megabyte = 1024 kilobyte and 1 kilobyte = 1024 byte so 100 megabyte = 100 * 1024 * 1024 bytes
#define size (1 *1024)

int main(void){
	// Step 2:
	unsigned char *buffer = (unsigned char*)big_random_block(size); // buffer is like an array that contains random characters
	
	// Step 3:
	unsigned int histo[256];
	for (int i = 0; i < 256; i++){
		histo[i] = 0;
	}

	// Step 4 and 5:
	for (int i = 0; i <size; i++){
		histo[buffer[i]]++;
		printf("buffer : %c and histo : %d\n", buffer[i], histo[buffer[i]]); // it is just to show the buffer
	}

	// Step 6:
	long histoCount = 0;
	for (int i = 0; i <256; i++){
		histoCount += histo[i];
	}

	printf("Histogram Sum : %ld\n", histoCount);

	// Step 7:
	free(buffer);
	return 0;
}
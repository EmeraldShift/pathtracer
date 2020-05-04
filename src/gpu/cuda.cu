#include "cuda.cuh"

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

void print_from_gpu(){
	//launches kernel
	cuda_hello<<<1,1>>>();
}

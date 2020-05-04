#include "../RayTracer.h"
#include "cuda.cuh"

void RayTracer::traceImageGPU(int w, int h) {
    //sets up buffers, updates ui parameters
    traceSetup(w, h);

    //copy pixel buffer onto device
    //cudaMalloc
    //cudaMemcpy

    //determine kenel paramaters
    int num_blocks = 1;
    int block_size = 1;

    //launch tracer kernels
    cuda_hello<<<num_blocks, block_size>>>();

    //sync
    cudaDeviceSynchronize();
}


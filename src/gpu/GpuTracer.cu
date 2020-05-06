#include "GpuTracer.h"

#include <iostream>

constexpr int THREADS_PER_BLOCK = 32;

CUDA_GLOBAL
static void tracePixel(double *buffer, unsigned width, unsigned height) {
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned x = id % width;
    unsigned y = id / width;
    if (id < width * height) {
        buffer[id * 3] = (double) x / width;
        buffer[id * 3 + 1] = (double) y / height;
        buffer[id * 3 + 2] = 0;
    }
}

void GpuTracer::traceImage(int width, int height) {
    auto size = width * height * 3;
    auto raw = new double[size];
    double *d_raw;
    cudaMalloc((void **) &d_raw, size * sizeof(double));

    cudaMemcpy(d_raw, raw, size * sizeof(double), cudaMemcpyHostToDevice);
    tracePixel<<<(size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_raw, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(raw, d_raw, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            setPixel(x, y, glm::dvec3(raw[(x+y*width)*3], raw[(x+y*width)*3+1], raw[(x+y*width)*3+2]));
}

void GpuTracer::waitRender() {
}
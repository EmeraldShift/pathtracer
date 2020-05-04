#ifndef __cuda_hello__
#define __cuda_hello__
#include <stdio.h>

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

__global__ void cuda_hello();
void print_from_gpu();
#endif

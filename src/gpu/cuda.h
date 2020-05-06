#pragma once

#include <stdio.h>

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_GLOBAL __global__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_GLOBAL
#endif

void print_from_gpu();

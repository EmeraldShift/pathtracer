#include "../RayTracer.h"
#include "cuda.h"


/*
    Note for synchronize(): This function waits for completion of all grids
    previously launched by the thread block *from which it has been called.* 
*/


__device__
void traceRayGPU(){
    printf("hello from the ray trace gpu code\n");


}

//shoots primary ray
__global__
void traceGPU(){

    traceRayGPU();

}



//anti-aliasing, multiple rays per pixel
__global__
void tracePixelGPU(){

    traceGPU();

}

__host__
void RayTracer::traceImageGPU(int w, int h) {
    //sets up buffers, updates ui parameters
    traceSetup(w, h);

    //convert pixel buffer into not glm::vec3?

    /*
        copy pixel buffer onto device 
        and other internal states referenced by tracePixel
    */

    //cudaMalloc
    //cudaMemcpy


    int num_blocks = 1;
    int block_size = 1;

    //launch tracer kernels per pixel
    //should take in pointer to pixel buffer
    tracePixelGPU<<<num_blocks, block_size>>>();

    //copy back pixel color result to host

    //sync
    cudaDeviceSynchronize();
}


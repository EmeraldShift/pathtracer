#include "../RayTracer.h"


/*
    Note for synchronize(): This function waits for completion of all grids
    previously launched by the thread block *from which it has been called.* 
*/


__global__
void traceRayGPU(){

    // int num_blocks = 1;
    // int block_size = 1;


}

//shoots primary ray
__global__
void traceGPU(){

    int num_blocks = 1;
    int block_size = 1;

    //traceRayGPU<<<num_blocks, block_size>>>();
    //cudaDeviceSynchronize();

}



//anti-aliasing, multiple rays per pixel
__global__
void tracePixelGPU(){

    int num_blocks = 1;
    int block_size = 1;

    //JK WE DONT HAVE THE ARCH TO SUPPORT THIS
    //traceGPU<<<num_blocks, block_size>>>();

    //can use the following to wait on children kernels
    //cudaDeviceSynchronize();


}

__host__
void RayTracer::traceImageGPU(int w, int h) {
    //sets up buffers, updates ui parameters
    traceSetup(w, h);

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


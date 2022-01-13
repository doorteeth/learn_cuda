#include "learnCuda.h"
#include <cuda_runtime.h>
#include <stdio.h>



int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);

    printf("Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("Warp size%d\n", deviceProp.warpSize);
    printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor: %d\n",
    deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n",
           deviceProp.maxThreadsPerMultiProcessor/32);
    return EXIT_SUCCESS;
}

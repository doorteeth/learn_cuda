#include <cuda_runtime.h>
#include <stdio.h>

#include "learnCuda.h"



void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx=0; idx<N; idx++)
        C[idx] = A[idx] + B[idx];
}
int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    // set matrix dimension
    int nx=8;
    int ny=6;
    int nxy=nx*ny;

    size_t nBytes = nxy * sizeof(float);

    //malloc host memory
    int *h_A = (int *) malloc(nBytes);
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    //malloc device memory
    int *d_A;
    cudaMalloc((void **)&d_A, nBytes);

    //transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    //set up execution configuration
    dim3 block(4,2);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);

    //invoke the kernel
    printThreadIndex<<<grid, block>>>(d_A, nx, ny);
    cudaDeviceSynchronize();

    //free host and device memory
    cudaFree(d_A);
    free(h_A);
    cudaDeviceReset();
    return(0);
}


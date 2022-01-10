#include <cuda_runtime.h>
#include <stdio.h>

#include "learnCuda.h"


void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; ++iy)
    {
        for (int ix = 0; ix < nx; ++ix)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    // set up date size of vectors
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    printf("Matrix size:nx %d ny %d\n", nx, ny);

    // malloc host memory
    size_t nBytes = nxy * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *) malloc(nBytes);
    h_B = (float *) malloc(nBytes);
    hostRef = (float *) malloc(nBytes);
    gpuRef = (float *) malloc(nBytes);
    double iStart, iElaps;
    // initialize data at host side
    iStart = cpuSecond();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    iElaps = cpuSecond() - iStart;
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    // add vector at host side for result checks
    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnHost Time elapsed %f" \
    "sec\n", iElaps);
    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **) &d_A, nBytes);
    cudaMalloc((float **) &d_B, nBytes);
    cudaMalloc((float **) &d_C, nBytes);
    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    // invoke kernel at host side
    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnGPU <<<(%d,%d), (%d,%d)>>> Time elapsed %f" \
    "sec\n", grid.x, grid.y, block.x, block.y, iElaps);
    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    // check device results
    checkResult(hostRef, gpuRef, nxy);
    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    return (0);
}

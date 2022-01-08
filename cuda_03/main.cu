#include <iostream>
#include <memory>

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int ) time(&t));

    for (int i = 0; i < size; ++i)
    {
        ip[i]=(float )(rand()&0xFF)/10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        C[i]=A[i]+B[i];
    }
}

__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        C[i]=A[i]+B[i];
    }
}

int main()
{
    std::cout << "Hello, World!" << std::endl;

    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);


    float *h_A, *h_B, *h_C, *h_out;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    h_out = (float *)malloc(nBytes);

    float *d_A, *d_B, *d_C;

    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);
//
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    sumArraysOnHost(h_A, h_B, h_C, nElem);
    sumArraysOnDevice<<<1,1>>>(d_A, d_B, d_C, nElem);


    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_C, nBytes, cudaMemcpyDeviceToHost);


    double epsilon=1.0E-8;

    for (int i = 0; i < nElem; ++i)
    {
        if (abs(h_out[i]-h_C[i])>epsilon)
        {
            std::cout<<"Failure"<<std::endl;
            return 1;
        }
    }
    std::cout<<"Success"<<std::endl;

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_out);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

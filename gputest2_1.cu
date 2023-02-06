#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <device_launch_parameters.h>

//CUBLAS matmul

static __inline__ void modify(cublasHandle_t handle, float (*A)[10000], float (*B)[10000], float (*C)[10000], \
const float a,  const float b, int n, int r, int c)
{
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &a, *A, n, *B, n,\
     &b, *C, n);
}

int main(void) {
    cublasStatus_t stat;
    cublasHandle_t handle;
    int M = 10000, N = 10000;

    float (*A)[10000] = new float [N][10000];
    float (*B)[10000] = new float [N][10000];
    float (*C)[10000] = new float [N][10000];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 2.0f + (i*0.01f);
            B[i][j] = 2.0f + (i*0.01f);
            C[i][j] = 6.0f + (i*0.01f);
        }
    }
    // Allocate 3 arrays on GPU
    float (*dA)[10000], (*dB)[10000], (*dC)[10000];
    size_t a = sizeof(float[10000][10000]);
    cudaMalloc(&dA, a);
    cudaMalloc(&dB, a);
    cudaMalloc(&dC, a);
    
    cudaMemcpy(dA, A, a,cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, a,cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, a,cudaMemcpyHostToDevice);
    stat = cublasCreate(&handle);
    modify (handle, dA, dB, dC, 1.0f, 1.0f, N, N, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, a,cudaMemcpyDeviceToHost);
    
    printf("hello this is test %f, %f", C[1][1], C[22][22]);
    // //Free GPU memory
    cudaFree(dA), cudaFree(dB), cudaFree(dC);
    
    // // Free CPU memory
    free(A), free(B), free(C);
 }
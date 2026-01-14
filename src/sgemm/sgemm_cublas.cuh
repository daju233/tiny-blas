#include <cublas_v2.h>
#include "../utils/common.h"

void sgemm_cublas(const float* hA, const float* hB, float* hC, int M, int N, int K, float* time) {
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    // Row-major C = A(MxK) * B(KxN) using column-major cublas:
    // Compute: C^T(NxM) = B^T(NxK) * A^T(KxM)
    // cublasSgemm: (opB, opA): N,M,K with column-major leading dims
    for (int i = 0; i < WARMLOOP; ++i) {
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, K,
            dA, M,
            &beta,
            dC, N);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < PERFLOOP; ++i) {
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, K,
            dA, M,
            &beta,
            dC, N);
    }
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
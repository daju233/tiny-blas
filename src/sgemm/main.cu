#include "my_sgemm.cuh"
#include "sgemm_cublas.cuh"
#include <vector>
#include <random>
#include "../utils/common.h"

#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_RESET "\033[0m"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    int M = 1024;
    int N = 1024;
    int K = 1024;

    std::vector<float> hA(M * K), hB(K * N), hC(M * N), hC_ref(M * N);

    for (int i = 0; i < M * K; ++i) hA[i] = i%2==0? 1 : 2;
    for (int i = 0; i < K * N; ++i) hB[i] = i%2==0? 2 : 1;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));

    float cublas_ms = 0.0f;
    sgemm_cublas(hA.data(), hB.data(), hC_ref.data(), M, N, K, &cublas_ms);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / (threadsPerBlock.x),
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int i = 0; i < WARMLOOP; ++i) {
        my_sgemm_v3<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, M, N, K);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < PERFLOOP; ++i) {
        my_sgemm_v3<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();

    float my_ms = 0.0f;
    cudaEventElapsedTime(&my_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_err = fmaxf(max_err, fabsf(hC[i] - hC_ref[i]));
        if (max_err > 1e-4f) {
            printf("error: C[%d]=%.6f, C_ref[%d]=%.6f\n", i, hC[i], i, hC_ref[i]);
            break;
        }
    }
    if (max_err <= 1e-4f) printf("kernel output all right\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    printf("cublas sgemm time: %.3fms\n", cublas_ms / PERFLOOP);
    printf("my sgemm time: %.3fms\n", my_ms / PERFLOOP);

    float pct = (cublas_ms / my_ms) * 100.0f;
    if (pct < 100.0f) {
        printf(COLOR_RED "my_sgemm is %.2f%% of cublas (slower)\n" COLOR_RESET, pct);
    } else {
        printf(COLOR_GREEN "my_sgemm is %.2f%% of cublas (faster)\n" COLOR_RESET, pct);
    }
    return 0;
}
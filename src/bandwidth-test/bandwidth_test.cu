// bandwidth_kernel.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void bandwidth_test(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx]; 
    }
}

int main() {
    const size_t N = 1 << 27; // 128M floats = 512 MB
    const size_t bytes = N * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // 初始化输入（防止零页优化）
    cudaMemset(d_in, 1, bytes);

    // Warm-up
    bandwidth_test<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 20; ++i) {
        bandwidth_test<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double sec = ms / 1000.0 / 20.0;

    // 每次调用：读 N floats + 写 N floats = 2 * N * 4 bytes
    double total_bytes = 2.0 * bytes;
    double bandwidth_gb_s = (total_bytes / 1e9) / sec;

    std::cout << "Effective Memory Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
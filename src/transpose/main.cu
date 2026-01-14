#include "my_transpose.cuh"
#include "transpose_cublas.cuh"
#include <vector>
#include <random>
#include "../utils/common.h"

#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_RESET "\033[0m"
// size_t free_mem, total_mem;


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(){
    std::mt19937 gen(42); // random number generator
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // int rows = 16384; 
    // int cols = 16384; 
    int rows = 1920; 
    int cols = 1080; 
    
    std::vector<float> h_input(rows * cols);
    std::vector<float> h_output(rows * cols);
    std::vector<float> h_output_cublas(rows * cols);

    float* d_input;
    float* d_output;

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            h_input[i * cols + j]= dist(gen);
        }
    }

    float cublas_milliseconds;
    transpose_cublas(h_input.data(), h_output_cublas.data(), rows, cols, &cublas_milliseconds);

    dim3 threadsPerBlock(32,32);
    dim3 blockPerGrid((cols+32-1) / 32,(rows+32-1) / 32); 
    // cudaMemGetInfo(&free_mem, &total_mem);
    // printf("Before cudaMalloc: Free memory: %zu bytes, Total memory: %zu bytes\n", free_mem, total_mem);

    CUDA_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, rows * cols * sizeof(float)));
    // cudaMemGetInfo(&free_mem, &total_mem);
    // printf("After cudaMalloc: Free memory: %zu bytes, Total memory: %zu bytes\n", free_mem, total_mem);

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, h_output.data(), rows * cols *sizeof(float), cudaMemcpyHostToDevice));

    for(int i=0; i< WARMLOOP; ++i){
        transpose_v3<<<blockPerGrid,threadsPerBlock>>>(d_input, d_output, rows, cols);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    for(int i=0; i< PERFLOOP; ++i){
        // transpose_v1<<<blockPerGrid,threadsPerBlock>>>(d_input, d_output, rows, cols);
        transpose_v3<<<blockPerGrid,threadsPerBlock>>>(d_input, d_output, rows, cols);
    }
    cudaEventRecord(stop,0);
    cudaDeviceSynchronize();

    float my_milliseconds = 0;
    cudaEventElapsedTime(&my_milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, rows * cols *sizeof(float), cudaMemcpyDeviceToHost));

    float error = 0;
    bool tmp = true;
    for (int i = 0; i < rows * cols; i++){
        // printf("cublas[%d] = %.3f\n",i, h_output_cublas[i]);
        error = fmax(error, fabs(h_output[i] - h_output_cublas[i]));
        if (error > 1e-5)
        {
            tmp = false;
            printf("error:hostC[%d] = %.3f, cublas[%d] = %.3f\n", i, h_output[i], i, h_output_cublas[i]);
            break;
        }
    }
    
    if (tmp)
    {
        printf("kernel output all right\n");
    }
    cudaFree(d_input);
    cudaFree(d_output);

    printf("transpose_cublas used time:%.3fms\n",cublas_milliseconds/PERFLOOP);
    printf("transpose_v3 used time:%.3fms\n",my_milliseconds/PERFLOOP);

    //compare percentage
    float percentage = (cublas_milliseconds / my_milliseconds) * 100;

    if (percentage < 100) {
        printf(COLOR_RED "transpose_v3 is %.2f%% of cublas version (slower)\n" COLOR_RESET, percentage);
    } else {
        printf(COLOR_GREEN "transpose_v3 is %.2f%% of cublas version (fasterer)" COLOR_RESET, percentage);

    }

}
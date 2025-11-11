#include "transpose_v1.cuh"
#include "transpose_cublas.cuh"
#include <vector>
#include <random>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(){
    std::mt19937 gen(42); // 初始化随机数生成器
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    int rows = 1000;
    int cols = 1000;
    
    std::vector<float> h_input(rows * cols);
    std::vector<float> h_output(rows * cols);
    std::vector<float> h_output_cublas(rows * cols);

    const float* d_input;
    float* d_output;

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            h_input[rows * i + j]= dist(gen);
        }
    }

    dim3 threadsPerBlock(32,32);
    dim3 blockPerGrid((rows+32-1) / 32,(cols+32-1) / 32);

    cudaMalloc((void **)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void **)&d_output, rows * cols * sizeof(float));

    cudaMemcpy((void *)d_input, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_output, h_output.data(), rows * cols *sizeof(float), cudaMemcpyHostToDevice);

    transpose_cublas(h_input.data(), h_output_cublas.data(), rows, cols);
    transpose_v1<<<threadsPerBlock,blockPerGrid>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output, rows * cols *sizeof(float), cudaMemcpyDeviceToHost);

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
        printf("cublas output all right\n");
    }
    cudaFree(&d_input);
    cudaFree(&d_output);

}
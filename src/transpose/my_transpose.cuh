#include "../utils/common.h"

__global__ void transpose_v1(const float* input, float* output, int rows, int cols){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if(row < rows && col < cols){
        output[col * rows + row] = input[cols * row + col];
    }
}

__global__ void transpose_v2(const float* input, float* output, int rows, int cols){

    __shared__ float tile[TILE_DIM][TILE_DIM];
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    tile[threadIdx.y][threadIdx.x] = input[cols * row + col];
    __syncthreads();
    
    col = threadIdx.x + blockDim.y * blockIdx.y;
    row = threadIdx.y + blockDim.x * blockIdx.x;

    if(row < rows && col < cols){
        output[cols * row + col] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_v3(const float* input, float* output, int rows, int cols){

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    tile[threadIdx.y][threadIdx.x] = input[cols * row + col];
    __syncthreads();
    
    col = threadIdx.x + blockDim.y * blockIdx.y;
    row = threadIdx.y + blockDim.x * blockIdx.x;

    if(row < rows && col < cols){
        output[cols * row + col] = tile[threadIdx.x][threadIdx.y];
    }
}
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

__global__ void transpose_v3(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int in_row = threadIdx.y + blockIdx.y * blockDim.y;
    int in_col = threadIdx.x + blockIdx.x * blockDim.x;

    tile[threadIdx.y][threadIdx.x] = input[in_row * cols + in_col];
    __syncthreads();

    int out_col = threadIdx.x + blockIdx.y * blockDim.y;//block用来定位，读取的smem本身就是已经转置好的，内部按行优先读就行了
    int out_row = threadIdx.y + blockIdx.x * blockDim.x;

    if (out_row < cols && out_col < rows) {
        output[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void transpose_v4_32x32(int M, int N,                  //
                                   const float *__restrict__ iA,  //
                                   float *__restrict__ oA) {
  __shared__ float smem[32][32];
  // Global Memory -> Shared Memory
  {
    constexpr const int ITER_Y = 32 / BLOCK_DIM_Y;//每个block 32
    constexpr const int ITER_X = 32 / BLOCK_DIM_X;
    static_assert(ITER_Y * BLOCK_DIM_Y == 32);
    static_assert(ITER_X * BLOCK_DIM_X == 32);

#pragma unroll
    for (int iy = 0; iy < ITER_Y; iy++) {
      const int ly = iy * BLOCK_DIM_Y + threadIdx.x / BLOCK_DIM_X;//
      const int gy = blockIdx.x * 32 + ly;//转置
#pragma unroll
      for (int ix = 0; ix < ITER_X; ix++) {
        const int lx = ix * BLOCK_DIM_X + threadIdx.x % BLOCK_DIM_X;
        const int gx = blockIdx.y * 32 + lx;
        if (gy < M && gx < N) {
          smem[lx][ly] = iA[gy * N + gx];
        }
      }
    }
  }
  __syncthreads();

  // Shared Memory -> Global Memory
  {
    constexpr const auto ITER_Y = 32 / BLOCK_DIM_Y;
    constexpr const auto ITER_X = 32 / BLOCK_DIM_X;
    static_assert(ITER_Y * BLOCK_DIM_Y == 32);
    static_assert(ITER_X * BLOCK_DIM_X == 32);

#pragma unroll
    for (int iy = 0; iy < ITER_Y; iy++) {
      const int ly = iy * BLOCK_DIM_Y + threadIdx.x / BLOCK_DIM_X;
      const int gy = blockIdx.y * 32 + ly;
#pragma unroll
      for (int ix = 0; ix < ITER_X; ix++) {
        const int lx = ix * BLOCK_DIM_X + threadIdx.x % BLOCK_DIM_X;
        const int gx = blockIdx.x * 32 + lx;
        if (gy < N && gx < M) {
          oA[gy * M + gx] = smem[ly][lx];//转换block的xy定位block位置，内部数据使用smem对调xy进行读取等于转置，加起来等于全局转置
        }
      }
    }
  }
}

__global__ void transpose_v5(const float* input, float* output, int rows, int cols) {
//TODO:进一步提升读写
}
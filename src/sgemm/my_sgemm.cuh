#include "../utils/common.h"

__global__ void my_sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K){
    int Row = threadIdx.y + blockDim.y * blockIdx.y;
    int Col = threadIdx.x + blockDim.x * blockIdx.x;
    if(Row < M && Col < N){
        float acc = 0.0f;
        for(int i = 0;i < K; ++i){
            acc += A[Row * K + i] * B[i * N + Col];
        }
        C[Row * N + Col] = acc;
    }
}

__global__ void my_sgemm_v2(const float* A, const float* B, float* C, int M, int N, int K){
    __shared__ float Ads[TILE_DIM][TILE_DIM];
    __shared__ float Bds[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;
    int Row = by * TILE_DIM + ty; int Col = bx * TILE_DIM + tx;

    float acc = 0;
    for(int phase = 0; phase < K/TILE_DIM; ++phase){
        Ads[ty][tx] = A[Row * K + phase * TILE_DIM + tx];
        Bds[ty][tx] = B[(phase * TILE_DIM + ty)*K + Col];
        __syncthreads();

        for(int k = 0; k < TILE_DIM; ++k){
            acc += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    C[Row * K + Col] = acc;
}

__global__ void my_sgemm_v3(const float* A, const float* B, float* C, int M, int N, int K){
    __shared__ float Ads[TILE_DIM][TILE_DIM + 1];
    __shared__ float Bds[TILE_DIM][TILE_DIM * COARSE_FACTOR + 1];

    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;
    int Row = by * TILE_DIM + ty;
    int colStart = bx*TILE_DIM*COARSE_FACTOR +tx;

    float acc[COARSE_FACTOR];
    for(int c = 0;c < COARSE_FACTOR;++c){
        acc[c] = 0.0f;
    }

    int numPhase = (K + TILE_DIM - 1) / TILE_DIM;

    for(int phase = 0; phase < numPhase; ++phase){
        int a_col = phase * TILE_DIM + tx;
        Ads[ty][tx] = (Row < M && a_col < K) ? A[Row * K + a_col] : 0.0f;

        int b_row = phase * TILE_DIM + ty;
        for(int c = 0;c < COARSE_FACTOR; ++c){
            int b_col = colStart + c * TILE_DIM;
            int smem_col = c * TILE_DIM + tx;
            Bds[ty][smem_col] = (b_row < K && b_col < N) ? B[b_row*N + b_col] : 0.0f;

        }
        __syncthreads();

        for(int k = 0; k < TILE_DIM; ++k){
            float a_val = Ads[ty][k];
            for(int c = 0; c < COARSE_FACTOR; ++c){
                acc[c] += a_val * Bds[k][c * TILE_DIM + tx];
            }
        }

        __syncthreads();
    }
    for(int c = 0; c < COARSE_FACTOR; ++c){
        int col  = colStart + c * TILE_DIM;
        if (Row < M && col < N){
            C[Row * N + col] = acc[c];
        }
    }
}
#include <cublas_v2.h>
#include<iostream>
#include "../utils/common.h"

void transpose_cublas(const float* input, float* output, int rows, int cols, float* time){
    float *dat, *da;
    int T = rows;
    // size_t free_mem, total_mem;

    // cudaMemGetInfo(&free_mem, &total_mem);
    // printf("Before cudaMalloc: Free memory: %zu bytes, Total memory: %zu bytes\n", free_mem, total_mem);
    cudaMalloc(&da,  rows*cols*sizeof(input[0]));
    cudaMalloc(&dat, rows*cols*sizeof(output[0]));
    // cudaMemGetInfo(&free_mem, &total_mem);
    // printf("After cudaMalloc: Free memory: %zu bytes, Total memory: %zu bytes\n", free_mem, total_mem);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMemcpy(da, input, rows*cols*sizeof(da[0]), cudaMemcpyHostToDevice);
  // tranpose(da) -> dat


for(int i=0; i< WARMLOOP; ++i){
    cublasSgeam(handle,
      CUBLAS_OP_T, CUBLAS_OP_T,
      T, cols,
      &alpha, da, cols,
      &beta, da, cols,
      dat, T
);
}

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start,0);  
  for(int i=0; i< PERFLOOP; ++i){
    cublasSgeam(handle,
      CUBLAS_OP_T, CUBLAS_OP_T,
      T, cols,
      &alpha, da, cols,
      &beta, da, cols,
      dat, T
);
}
cudaEventRecord(stop,0);
cudaDeviceSynchronize();

cudaEventElapsedTime(time, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);

cudaMemcpy(output, dat, rows * cols *sizeof(float), cudaMemcpyDeviceToHost);
// cudaMemGetInfo(&free_mem, &total_mem);
// printf("Before cudaFree: Free memory: %zu bytes, Total memory: %zu bytes\n", free_mem, total_mem);
cudaFree(da);//注意 传入设备指针
cudaFree(dat);
// cudaMemGetInfo(&free_mem, &total_mem);
// printf("After cudaFree: Free memory: %zu bytes, Total memory: %zu bytes\n", free_mem, total_mem);
}
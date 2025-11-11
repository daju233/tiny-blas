#include <cublas_v2.h>

void transpose_cublas(const float* input, float* output, int rows, int cols){
    float *dat, *da;
    int T = rows;
    cudaMalloc(&da,  rows*cols*sizeof(input[0]));
    cudaMalloc(&dat, rows*cols*sizeof(output[0]));
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMemcpy(da, input, rows*cols*sizeof(da[0]), cudaMemcpyHostToDevice);
  // tranpose(da) -> dat
    cublasSgeam(handle,
                  CUBLAS_OP_T, CUBLAS_OP_T,
                  T, cols,
                  &alpha, da, cols,
                  &beta, da, cols,
                  dat, T
      );
    cudaMemcpy(output, dat, rows * cols *sizeof(float), cudaMemcpyDeviceToHost);
}
__global__ void transpose_v1(const float* input, float* output, int rows, int cols){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if(row < rows && col < cols){
        output[col * rows + row] = input[cols * row + col];
    }
}
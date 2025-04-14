#include <cuda_runtime.h>
#include <stdio.h>
#include "cross_correlation.cuh"

__constant__ float kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void cross_correlation_2d(const float* input, float* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row

    if (x < OUTPUT_MATRIX_SIZE && y < OUTPUT_MATRIX_SIZE) {
        float sum = 0.0f;
        for (int m = 0; m < KERNEL_SIZE; ++m) {
            for (int n = 0; n < KERNEL_SIZE; ++n) {
                int input_row = y + m;
                int input_col = x + n;
                float input_val = input[input_row * INPUT_MATRIX_SIZE + input_col]; 
                float kernel_val = kernel[m * KERNEL_SIZE + n];
                sum += input_val * kernel_val;
            }
        }
        output[y * OUTPUT_MATRIX_SIZE + x] = sum;
    }
}

void run_cross_correlation(const float* h_input, const float* h_kernel, float* h_output, float& elapsed_time_ms) {
    float *d_input, *d_output;

    cudaMalloc(&d_input, INPUT_MATRIX_SIZE * INPUT_MATRIX_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_MATRIX_SIZE * OUTPUT_MATRIX_SIZE * sizeof(float));
    cudaMemcpy(d_input, h_input, INPUT_MATRIX_SIZE * INPUT_MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((OUTPUT_MATRIX_SIZE + 15) / 16, (OUTPUT_MATRIX_SIZE + 15) / 16);
    cross_correlation_2d<<<numBlocks, threadsPerBlock>>>(d_input, d_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    cudaMemcpy(h_output, d_output, OUTPUT_MATRIX_SIZE * OUTPUT_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

#include <stdio.h>
#include <cuda_runtime.h>

#define INPUT_MATRIX_SIZE 256  // 256x256 input matrix
#define KERNEL_SIZE 8          // 8x8 kernel 
#define OUTPUT_MATRIX_SIZE (INPUT_MATRIX_SIZE - KERNEL_SIZE + 1) // output matrix

__constant__ int kernel[KERNEL_SIZE * KERNEL_SIZE]; // creates kernel

__global__ void cross_correlation_2d(float* input, float* output) {

    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row

    if (x < OUTPUT_MATRIX_SIZE && y < OUTPUT_MATRIX_SIZE) {
        float sum = 0.0f;
        // Correlation Calculation
        for (int m = 0; m < KERNEL_SIZE; ++m) {
            for (int n = 0; n < KERNEL_SIZE; ++n) {
                int input_row = y + m;
                int input_col = x + n;
                float input_val = input[input_row * INPUT_MATRIX_SIZE + input_col]; 
                float kernel_val = kernel[m * KERNEL_SIZE + n];
                sum += input_val * kernel_val;
            }
        }
        // output matrix
        output[y * OUTPUT_MATRIX_SIZE + x] = sum;
    }
}

int main() {
    // Host matrices
    float h_input[INPUT_MATRIX_SIZE][INPUT_MATRIX_SIZE], h_kernel[KERNEL_SIZE][KERNEL_SIZE], h_output[OUTPUT_MATRIX_SIZE][OUTPUT_MATRIX_SIZE];
    float *d_input, *d_output;

    // Initialize matrices with random sample values
    for (int i = 0; i < INPUT_MATRIX_SIZE; i++) {
        for (int j = 0; j < INPUT_MATRIX_SIZE; j++) {
            h_input[i][j] = (2.0f * rand()/RAND_MAX - 1.0f);  // Sample input
        }
    }

    for (int i = 0; i < KERNEL_SIZE; i++){
        for (int j = 0; j < KERNEL_SIZE; j++){
            h_kernel[i][j] = (2.0f * rand()/RAND_MAX - 1.0f);  // Sample kernel
        }
    }
    

    // Allocate device memory
    cudaMalloc(&d_input, INPUT_MATRIX_SIZE * INPUT_MATRIX_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_MATRIX_SIZE * OUTPUT_MATRIX_SIZE * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, INPUT_MATRIX_SIZE * INPUT_MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy Const kernel to device
    cudaMemcpyToSymbol(kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE*sizeof(float));

    // Create Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start Measuring
    cudaEventRecord(start);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 Blocks((OUTPUT_MATRIX_SIZE + 15) / 16, (OUTPUT_MATRIX_SIZE + 15) / 16);
    cross_correlation_2d<<<Blocks, threadsPerBlock>>>(d_input, d_output);

    // Stop Measuring 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate Elapsed Time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output Time Duration
    printf("TIME ELAPSED: %f MILLISECONDS\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, INPUT_MATRIX_SIZE * INPUT_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(kernel);
    cudaFree(d_output);

    printf("COMPLETED SUCCESSFULLY!\n");

    return 0;
}

#ifndef CROSS_CORRELATION_CUH
#define CROSS_CORRELATION_CUH

#define INPUT_MATRIX_SIZE 256
#define KERNEL_SIZE 8
#define OUTPUT_MATRIX_SIZE (INPUT_MATRIX_SIZE - KERNEL_SIZE + 1)

void run_cross_correlation(const float* h_input, const float* h_kernel, float* h_output, float& elapsed_time_ms);

#endif

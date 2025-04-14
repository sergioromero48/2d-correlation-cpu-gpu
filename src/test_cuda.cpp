#include <iostream>
#include <cstdlib>
#include "cross_correlation.cuh"

int main() {
    float input[INPUT_MATRIX_SIZE * INPUT_MATRIX_SIZE];
    float kernel[KERNEL_SIZE * KERNEL_SIZE];
    float output[OUTPUT_MATRIX_SIZE * OUTPUT_MATRIX_SIZE];
    float elapsed;

    // Fill input and kernel with random values
    for (int i = 0; i < INPUT_MATRIX_SIZE * INPUT_MATRIX_SIZE; ++i)
        input[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i)
        kernel[i] = 2.0f * rand() / RAND_MAX - 1.0f;

    run_cross_correlation(input, kernel, output, elapsed);

    std::cout << "TIME ELAPSED: " << elapsed << " ms\n";
    std::cout << "COMPLETED SUCCESSFULLY!\n";
    return 0;
}

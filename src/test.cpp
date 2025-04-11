#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "cross_correlation.hpp"

// For brevity, we use the namespace from our library.
using namespace cross_corr;
using Matrix = std::vector<std::vector<float>>;

int main() {
    const int input_rows = 256;
    const int input_cols = 256;
    const int kernel_size = 8;
    const int num_threads = 4;

    // Set up random number generation for values between -1 and 1.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create input matrix.
    Matrix input(input_rows, std::vector<float>(input_cols));
    for (int i = 0; i < input_rows; ++i)
        for (int j = 0; j < input_cols; ++j)
            input[i][j] = dist(gen);

    // Create kernel matrix.
    Matrix kernel(kernel_size, std::vector<float>(kernel_size));
    for (int i = 0; i < kernel_size; ++i)
        for (int j = 0; j < kernel_size; ++j)
            kernel[i][j] = dist(gen);

    // Prepare the output matrix.
    Matrix output;

    // Start the timer.
    auto start = std::chrono::high_resolution_clock::now();

    // Perform threaded cross-correlation.
    cross_correlation_2d_threaded(input, kernel, output, num_threads);

    // Stop the timer.
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time in milliseconds.
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Print the elapsed time.
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

    // Print a small portion of the output to verify correctness.
    std::cout << "Output (5x5 top-left block):\n";
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j)
            std::cout << output[i][j] << "\t";
        std::cout << "\n";
    }

    return 0;
}

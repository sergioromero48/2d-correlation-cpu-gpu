#ifndef CROSS_CORRELATION_HPP
#define CROSS_CORRELATION_HPP

#include <vector>
#include <queue>
#include <thread>
#include <mutex>

// Wrap your functions in a namespace for clarity.
namespace cross_corr {
    /*
    -------------
    | IMPORTANT! |
    -------------
    */
    // ---------------------------------------------
    using Matrix = std::vector<std::vector<float>>;
    // ---------------------------------------------

    // Worker function that each thread will run.
    // This is not meant to be called directly by the user.
    void worker(const Matrix& input,
                const Matrix& kernel,
                Matrix& output,
                std::queue<std::pair<int, int>>& jobs,
                std::mutex& lock);

    // This function performs 2D cross-correlation using multiple threads.
    // It fills 'output' with the result of applying 'kernel' to 'input'.
    void cross_correlation_2d_threaded(const Matrix& input,
                                       const Matrix& kernel,
                                       Matrix& output,
                                       int num_threads);
}

#endif // CROSS_CORRELATION_HPP

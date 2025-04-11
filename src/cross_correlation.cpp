#include "cross_correlation.hpp"

namespace cross_corr {

    void worker(const Matrix& input,
                const Matrix& kernel,
                Matrix& output,
                std::queue<std::pair<int, int>>& jobs,
                std::mutex& lock) {
        int k_rows = kernel.size();
        int k_cols = kernel[0].size();

        while (true) {
            // holds job thread is currently working on
            std::pair<int, int> job;

            // lock queue access for memory safety
            lock.lock();

            if (jobs.empty()) {
                lock.unlock(); // unlock memory before exit
                return; // if job list empty, return
            }
            job = jobs.front(); // get next available job
            jobs.pop();
            lock.unlock(); // unlock queue so other threads can access data

            int i = job.first;
            int j = job.second;

            // cross-correlation computation of patch
            float sum = 0.0f;
            for (int m = 0; m < k_rows; ++m) {
                for (int n = 0; n < k_cols; ++n) {
                    sum += input[i + m][j + n] * kernel[m][n];
                }
            }
            // fill in output matrix
            output[i][j] = sum;
        }
    }

    void cross_correlation_2d_threaded(const Matrix& input,
                                       const Matrix& kernel,
                                       Matrix& output,
                                       int num_threads) {
        // derived parameters from data
        int input_rows = input.size();
        int input_cols = input[0].size();
        int kernel_rows = kernel.size();
        int kernel_cols = kernel[0].size();

        int output_rows = input_rows - kernel_rows + 1;
        int output_cols = input_cols - kernel_cols + 1;
    
        // resize output based on input size
        output.resize(output_rows, std::vector<float>(output_cols, 0.0f));

        // initialize job queue and thread lock
        std::queue<std::pair<int, int>> jobs;
        std::mutex lock;

        // fill job queue with output coordinates
        for (int i = 0; i < output_rows; ++i)
            for (int j = 0; j < output_cols; ++j)
                jobs.push({i, j});

        // launch threads
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back(worker,      // calls worker function
                                 std::ref(input),  // reference to input matrix
                                 std::ref(kernel), // reference to kernel
                                 std::ref(output), // reference to output matrix
                                 std::ref(jobs),   // reference to job queue
                                 std::ref(lock));  // reference to global lock for memory safety
        }

        // join threads (wait for all threads to finish)
        for (int i = 0; i < threads.size(); i++) {
            threads[i].join();
        }
    }

} // end namespace cross_corr

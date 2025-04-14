# Cross Correlation 2D Library

A high-performance C++ library for performing 2D cross-correlation on matrices using:

- **Multi-threaded CPU** (parallelized with `<thread>`)
- **GPU-accelerated CUDA** (offloads computation to NVIDIA GPUs)

## Overview

Cross-correlation slides a filter kernel over an input matrix and computes the dot product at each position. This library provides two optimized implementations for high-performance computing:

1. **Multi-threaded CPU**: Uses multiple threads to divide and accelerate the computation.
2. **GPU-accelerated CUDA**: Uses CUDA to leverage the massive parallelism of NVIDIA GPUs.

## Features

- Two high-performance implementations:
  - `cross_correlation_2d_threaded(input, kernel, output, num_threads)`
  - `cross_correlation_2d_cuda(input, kernel, output)`
- Unified API design for easy swapping of backend
- Alias: `using Matrix = std::vector<std::vector<float>>;`

## Requirements

- **C++ Compiler**: C++11 or newer
- **CUDA Toolkit**: v10.0+ and NVIDIA GPU for CUDA version
- **Operating System**: Linux/macOS (POSIX threading)

## Compilation

### Multi-threaded CPU Version

Files:
- `cross_correlation.hpp`
- `cross_correlation.cpp`
- `main.cpp`

Compile:

```bash
g++ -std=c++11 -pthread main.cpp cross_correlation.cpp -o cross_corr_cpu

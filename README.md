# Cross Correlation 2D Library

A versatile C++ library for performing 2D cross-correlation on matrices, with three implementations:

- **Single-threaded** (basic, no dependencies beyond the C++ standard library)
- **Multi-threaded** (parallel CPU version using `<thread>` and `<mutex>`)
- **GPU-accelerated** (CUDA implementation for NVIDIA GPUs)

## Overview

Cross-correlation slides a filter kernel over an input matrix and computes the dot product at each position. This library offers three implementations to suit different performance needs and environments:

1. **Single-threaded**: Simple, minimal overhead. Good for small matrices or debugging.
2. **Multi-threaded**: Uses POSIX threads to split work across CPU cores for faster execution on multi-core systems.
3. **GPU-accelerated**: Leverages CUDA to offload computation to an NVIDIA GPU, ideal for large matrices and high throughput.

## Features

- **Three implementations** in one library:
  - `cross_correlation_2d` (single-threaded)
  - `cross_correlation_2d_threaded` (multi-threaded)
  - `cross_correlation_2d_cuda` (GPU/CUDA)
- **Simple API**: All versions share the same function signature, making it easy to switch between implementations.
- **Matrix alias**: `using Matrix = std::vector<std::vector<float>>;`
- **Header-only definitions** for CPU versions; separate `.cu` file for CUDA.

## Requirements

- **C++ Compiler**: Supports C++11 or higher (e.g., Apple Clang, GCC).
- **CUDA Toolkit** (for GPU version): NVIDIA GPU and CUDA 10.0 or later.
- **Platform**: POSIX-compliant OS for CPU threading; NVIDIA drivers for CUDA.

## Installation and Compilation

### CPU Versions (Single- & Multi-threaded)

Assuming you have:
- `cross_correlation.hpp`
- `cross_correlation.cpp`
- `main.cpp` (sample usage)

Compile both CPU versions together:

```bash
g++ -std=c++11 -pthread main.cpp cross_correlation.cpp -o cross_corr_cpu
```

### GPU Version (CUDA)

Assuming you have:
- `cross_correlation.hpp` (CPU & CUDA declarations)
- `cross_correlation_cuda.cu` (CUDA implementation)
- `main_cuda.cpp` (sample usage)

Compile the CUDA version:

```bash
nvcc -std=c++11 main_cuda.cpp cross_correlation_cuda.cu -o cross_corr_gpu
```

## Usage Examples

Below are minimal examples for each implementation.

### Single-threaded

```cpp
#include "cross_correlation.hpp"
using namespace cross_corr;
using Matrix = std::vector<std::vector<float>>;

int main() {
    Matrix input = /* ... */;
    Matrix kernel = /* ... */;
    Matrix output;
    cross_correlation_2d(input, kernel, output);
    return 0;
}
```

### Multi-threaded

```cpp
#include "cross_correlation.hpp"
using namespace cross_corr;
using Matrix = std::vector<std::vector<float>>;

int main() {
    Matrix input = /* ... */;
    Matrix kernel = /* ... */;
    Matrix output;
    int threads = 4;
    cross_correlation_2d_threaded(input, kernel, output, threads);
    return 0;
}
```

### GPU-accelerated (CUDA)

```cpp
#include "cross_correlation.hpp"
using namespace cross_corr;
using Matrix = std::vector<std::vector<float>>;

int main() {
    Matrix input = /* ... */;
    Matrix kernel = /* ... */;
    Matrix output;
    cross_correlation_2d_cuda(input, kernel, output);
    return 0;
}
```

## Benchmarking

You can measure execution time with `<chrono>`:

```cpp
auto start = std::chrono::high_resolution_clock::now();
// call function
auto end = std::chrono::high_resolution_clock::now();
std::cout << "Elapsed ms: "
          << std::chrono::duration<double, std::milli>(end - start).count()
          << std::endl;
```

## License

[Your preferred license here]

## Contributing

Feel free to open issues or pull requests for bug fixes, performance improvements, or new features.

---


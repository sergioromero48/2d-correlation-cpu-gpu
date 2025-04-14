# Makefile for Cross Correlation 2D Library (CPU + CUDA)

CXX := g++
NVCC := nvcc

CXXFLAGS := -std=c++11 -O2 -pthread
NVCCFLAGS := -std=c++11 -O2

INCLUDE := -Isrc

CPU_SRC := src/cross_correlation.cpp src/main.cpp
CUDA_SRC := src/cross_correlation_cuda.cu src/main_cuda.cpp

CPU_BIN := cross_corr_cpu
CUDA_BIN := cross_corr_gpu

all: cpu cuda

cpu: $(CPU_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(CPU_SRC) -o $(CPU_BIN)

cuda: $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $(CUDA_SRC) -o $(CUDA_BIN)

clean:
	rm -f $(CPU_BIN) $(CUDA_BIN)

.PHONY: all cpu cuda clean

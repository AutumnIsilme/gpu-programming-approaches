
BLOCK_SIZE_X?=64
BLOCK_SIZE_Y?=64

CFLAGS=-O3 -std=c++17 -DBLOCK_SIZE_X=$(BLOCK_SIZE_X) -DBLOCK_SIZE_Y=$(BLOCK_SIZE_Y)
CXXFLAGS=-march=native
NVCCFLAGS=-arch=sm_86 -ccbin=clang++ -Xcompiler='-march=native -O3 -std=c++17'
LDFLAGS=

all: cuda base

cuda: stencil-2d-cuda.cpp stencil-2d-util.h
	nvcc -c $(CFLAGS) $(NVCCFLAGS) stencil-2d-cuda.cpp -o stencil-2d-cuda-cpp.o
	nvcc -c $(CFLAGS) $(NVCCFLAGS) stencil-2d-cuda.cu -o stencil-2d-cuda-cu.o
	nvcc $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) stencil-2d-cuda-cpp.o stencil-2d-cuda-cu.o -o stencil-2d-cuda

base: stencil-2d-base.cpp stencil-2d-util.h
	clang++ $(CFLAGS) $(CXXFLAGS) stencil-2d-base.cpp -o stencil-2d-base

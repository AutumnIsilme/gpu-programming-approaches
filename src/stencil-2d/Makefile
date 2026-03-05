
NVCC?=nvcc
NVCXX?=nvc++
CC?=gcc
CXX?=g++

BLOCK_SIZE_X?=16
BLOCK_SIZE_Y?=16

CFLAGS=-O3 -std=c++17 -DBLOCK_SIZE_X=$(BLOCK_SIZE_X) -DBLOCK_SIZE_Y=$(BLOCK_SIZE_Y)
CXXFLAGS=-mcpu=native
NVCCFLAGS=-arch=sm_86 -ccbin=$(CXX) -Xcompiler='$(CFLAGS) $(CXXFLAGS)'
LDFLAGS=

all: cuda base

cuda: stencil-2d-cuda.cpp stencil-2d-util.h
	$(NVCC) -c $(CFLAGS) $(NVCCFLAGS) stencil-2d-cuda.cpp -o stencil-2d-cuda-cpp.o
	$(NVCC) -c $(CFLAGS) $(NVCCFLAGS) stencil-2d-cuda.cu -o stencil-2d-cuda-cu.o
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) stencil-2d-cuda-cpp.o stencil-2d-cuda-cu.o -o stencil-2d-cuda

base: stencil-2d-base.cpp stencil-2d-util.h
	$(CXX) $(CFLAGS) $(CXXFLAGS) stencil-2d-base.cpp -o stencil-2d-base

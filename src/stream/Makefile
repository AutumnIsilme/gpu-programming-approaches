
NVCC?=nvcc
NVCXX?=nvc++
CC?=gcc
CXX?=g++

CFLAGS=-O3 -std=c++17
CXXFLAGS=-mcpu=native
NVCCFLAGS=-arch=sm_86 -ccbin=$(CXX) -Xcompiler='$(CFLAGS) $(CXXFLAGS)'
LDFLAGS=

all: cuda base cuda-b

cuda: stream-cuda.cpp stream-util.h
	$(NVCC) -c $(CFLAGS) $(NVCCFLAGS) stream-cuda.cpp -o stream-cuda-cpp.o
	$(NVCC) -c $(CFLAGS) $(NVCCFLAGS) stream-cuda.cu -o stream-cuda-cu.o
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) stream-cuda-cpp.o stream-cuda-cu.o -o stream-cuda

omp: stream-omp.cpp
	$(NVCXX) $(CFLAGS) $(CXXFLAGS) -mp=gpu -target=gpu stream-omp.cpp -o stream-omp

cuda-b: stream-cuda-b.cu
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) stream-cuda-b.cu -o stream-cuda-b

base: stream-base.cpp stream-util.h
	$(CXX) $(CFLAGS) $(CXXFLAGS) stream-base.cpp -o stream-base

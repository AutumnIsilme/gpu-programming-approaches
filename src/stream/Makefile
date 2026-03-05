
CFLAGS=-O3 -std=c++17
CXXFLAGS=-march=native
NVCCFLAGS=-arch=sm_86 -ccbin=clang++ -Xcompiler='-march=native -O3 -std=c++17'
LDFLAGS=

all: cuda base cuda-b

cuda: stream-cuda.cpp stream-util.h
	nvcc -c $(CFLAGS) $(NVCCFLAGS) stream-cuda.cpp -o stream-cuda-cpp.o
	nvcc -c $(CFLAGS) $(NVCCFLAGS) stream-cuda.cu -o stream-cuda-cu.o
	nvcc $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) stream-cuda-cpp.o stream-cuda-cu.o -o stream-cuda

omp: stream-omp.cpp
	nvc++ $(CFLAGS) $(CXXFLAGS) -mp=gpu -target=gpu stream-omp.cpp -o stream-omp

cuda-b: stream-cuda-b.cu
	nvcc $(CFLAGS) $(NVCCFLAGS) stream-cuda-b.cu -o stream-cuda-b

base: stream-base.cpp stream-util.h
	clang++ $(CFLAGS) $(CXXFLAGS) stream-base.cpp -o stream-base

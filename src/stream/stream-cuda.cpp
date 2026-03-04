#include "stream-util.h"
#include "stream-cuda.h"

#define BLOCK_SIZE 512

int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    size_t bytes = sizeof(double) * nx;
    double *dest;
    cudaMallocHost((void**)&dest, bytes);
    double *src;
    cudaMallocHost((void**)&src, bytes);

    double *_k_dest;
    cudaMalloc((void**)&_k_dest, bytes);

    double *_k_src;
    cudaMalloc((void**)&_k_src, bytes);


    // init
    initStream(dest, src, nx);
    cudaMemcpy(_k_dest, dest, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(_k_src, src, bytes, cudaMemcpyHostToDevice);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream(_k_src, _k_dest, nx, BLOCK_SIZE);
        std::swap(_k_src, _k_dest);
    }
    
    cudaDeviceSynchronize();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream(_k_src, _k_dest, nx, BLOCK_SIZE);
        std::swap(_k_src, _k_dest);
    }
    
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
     
    cudaMemcpy(dest, _k_dest, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(src, _k_src, bytes, cudaMemcpyDeviceToHost);

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    // check solution
    checkSolutionStream(dest, src, nx, nIt + nItWarmUp);
    
    // free memory
    cudaFree(dest);
    cudaFree(src);
    cudaFree(_k_dest);
    cudaFree(_k_src);

    return 0;
}

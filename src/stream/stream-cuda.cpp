#include "stream-util.h"
#include "stream-cuda.h"

#define BLOCK_SIZE 64

int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    const size_t bytes = sizeof(double) * nx;
    double *dest;
    cudaMallocHost((void**)&dest, bytes);
    double *src;
    cudaMallocHost((void**)&src, bytes);

    double *_kdest;
    cudaMalloc((void**)&_kdest, bytes);
    double *_ksrc;
    cudaMalloc((void**)&_ksrc, bytes);

    const size_t blocks = ceilingDivide(nx, BLOCK_SIZE);
    // init
    initStream(dest, src, nx);
    cudaMemcpy(_kdest, dest, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(_ksrc, src, bytes, cudaMemcpyHostToDevice);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream(_ksrc, _kdest, nx, blocks, BLOCK_SIZE);
        std::swap(_ksrc, _kdest);
    }
    
    cudaDeviceSynchronize();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream(_ksrc, _kdest, nx, blocks, BLOCK_SIZE);
        std::swap(_ksrc, _kdest);
    }
    
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
       
    cudaMemcpy(dest, _kdest, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(src, _ksrc, bytes, cudaMemcpyDeviceToHost);

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    // check solution
    checkSolutionStream(dest, src, nx, nIt + nItWarmUp);
    
    // free memory
    cudaFree(dest);
    cudaFree(src);
    cudaFree(_kdest);
    cudaFree(_ksrc);

    return 0;
}

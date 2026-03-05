#include "stencil-2d-util.h"
#include "stencil-2d-cuda.h"
#include "../cuda-util.h"

int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    //double *u;
    //u = new double[nx * ny];
    //double *uNew;
    //uNew = new double[nx * ny];

    const size_t bytes = sizeof(double) * nx * ny;
    double *u;
    cudaMallocHost((void**)&u, bytes);
    double *uNew;
    cudaMallocHost((void**)&uNew, bytes);

    double *_ku;
    cudaMalloc((void**)&_ku, bytes);
    double *_kuNew;
    cudaMalloc((void**)&_kuNew, bytes);

    const size_t blocks_x = ceilingDivide(nx, BLOCK_SIZE_X);
    const size_t blocks_y = ceilingDivide(ny, BLOCK_SIZE_Y);

    // init
    initStencil2D(u, uNew, nx, ny);

    cudaMemcpy(_ku, u, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(_kuNew, uNew, bytes, cudaMemcpyHostToDevice);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil2d(_ku, _kuNew, nx, ny, blocks_x, blocks_y, BLOCK_SIZE_X, BLOCK_SIZE_Y);
        std::swap(_ku, _kuNew);
        checkCudaError(cudaDeviceSynchronize(), true);
    }
    //checkCudaError(cudaDeviceSynchronize());

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil2d(_ku, _kuNew, nx, ny, blocks_x, blocks_y, BLOCK_SIZE_X, BLOCK_SIZE_Y);
        std::swap(_ku, _kuNew);
        checkCudaError(cudaDeviceSynchronize(), true);
    }
    //checkCudaError(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double), 7);

    // check solution
    checkCudaError(cudaMemcpy(u, _ku, bytes, cudaMemcpyDeviceToHost), true);
    checkCudaError(cudaMemcpy(uNew, _kuNew, bytes, cudaMemcpyDeviceToHost), true);

    checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp);

    cudaFree(u);
    cudaFree(uNew);
    cudaFree(_ku);
    cudaFree(_kuNew);

    return 0;
}

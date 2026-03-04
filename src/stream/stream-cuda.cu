//#include "stream-util.h"
#include "stream-cuda.h"

__global__ void _k_stream(const double *const __restrict__ src, double *__restrict__ dest, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i0 < nx) {
        dest[i0] = src[i0] + 1;
    }
}

void stream(const double *const __restrict__ src, double *__restrict__ dest, size_t nx, size_t block_size) {
    _k_stream<<<ceilingDivide(nx, block_size), block_size>>>(src, dest, nx);
}

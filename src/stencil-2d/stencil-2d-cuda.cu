#include "stencil-2d-cuda.h"

__global__ void _kstencil2d(const double *const __restrict__ u, double *__restrict__ uNew, size_t nx, size_t ny) {
    size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 > 0 && i0 < ny - 1 && i1 > 0 && i1 < nx - 1) {
        uNew[i0 + i1 * nx] = 0.25 * u[i0 + i1 * nx + 1] + 0.25 * u[i0 + i1 * nx - 1] + 0.25 * u[i0 + nx * (i1 + 1)] + 0.25 * u[i0 + nx * (i1 - 1)];
    }
}

inline void _hstencil2d(const double *const __restrict__ u, double *__restrict__ uNew, size_t nx, size_t ny, size_t blockIdxx, size_t blockIdxy, size_t blockDimx, size_t blockDimy, size_t threadIdxx, size_t threadIdxy) {
    size_t i0 = blockIdxx * blockDimx + threadIdxx;
    size_t i1 = blockIdxy * blockDimy + threadIdxy;

    if (i0 > 0 && i0 < ny - 1 && i1 > 0 && i1 < nx - 1) {
        uNew[i0 + i1 * nx] = 0.25 * u[i0 + i1 * nx + 1] + 0.25 * u[i0 + i1 * nx - 1] + 0.25 * u[i0 + nx * (i1 + 1)] + 0.25 * u[i0 + nx * (i1 - 1)];
    }
}


void stencil2d(const double *const __restrict__ u, double *__restrict__ uNew, size_t nx, size_t ny, size_t blocks_x, size_t blocks_y, size_t block_size_x, size_t block_size_y) {
    _kstencil2d<<<dim3(blocks_x, blocks_y), dim3(block_size_x, block_size_y)>>>(u, uNew, nx, ny);
    /*for (int i1 = 0; i1 < blocks_x; i1++)
        for (int i2 = 0; i2 < blocks_y; i2++)
            for (int i3=0; i3 < block_size_x; i3++)
                for (int i4=0; i4 < block_size_y; i4++)
                    _hstencil2d(u, uNew, nx, ny, i1, i2, block_size_x, block_size_y, i3, i4);*/
}



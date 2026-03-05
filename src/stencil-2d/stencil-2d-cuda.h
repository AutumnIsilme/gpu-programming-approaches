#pragma once

#include <cuda_runtime.h>

void stencil2d(const double *const __restrict__ u, double *__restrict__ uNew, size_t nx, size_t ny, size_t blocks_x, size_t blocks_y, size_t block_size_x, size_t block_size_y);

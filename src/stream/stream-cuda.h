#pragma once

#include <cuda_runtime.h>

void stream(const double *const __restrict__ src, double *__restrict__ dest, size_t nx, size_t block_size);
size_t ceilingDivide(size_t a, size_t b);

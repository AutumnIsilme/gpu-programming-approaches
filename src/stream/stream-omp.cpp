#include "stream-util.h"


inline void stream(const double *const __restrict__ src, double *__restrict__ dest, size_t nx) {
    #pragma omp target teams distribute parallel for
    for (size_t i0 = 0; i0 < nx; ++i0) {
        dest[i0] = src[i0] + 1;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    double *dest;
    dest = new double[nx];
    double *src;
    src = new double[nx];

    // init
    initStream(dest, src, nx);

    auto copystart = std::chrono::steady_clock::now();
    #pragma omp target enter data map(to: dest[0:nx], src[0:nx])
    auto copyend = std::chrono::steady_clock::now();
    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream(src, dest, nx);
        std::swap(src, dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream(src, dest, nx);
        std::swap(src, dest);
    }

    #pragma omp target exit data map(from: dest[0:nx], src[0:nx])
    
    auto end = std::chrono::steady_clock::now();

    printStats(end - start + copyend - copystart, nIt, nx, sizeof(double) + sizeof(double), 1);

    // check solution
    checkSolutionStream(dest, src, nx, nIt + nItWarmUp);

    delete[] dest;
    delete[] src;

    return 0;
}

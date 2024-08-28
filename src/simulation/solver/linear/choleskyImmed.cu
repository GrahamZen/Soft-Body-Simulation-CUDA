#include <linear/choleskyImmed.h>
#include "linearUtils.cuh"

template<>
void CholeskySpImmedSolver<double>::Solve(int N, double* d_b, double* d_x, double* A, int nz, int* rowIdx, int* colIdx, double* d_guess)
{
    int singularity;
    sort_coo(N, nz, A, rowIdx, colIdx, d_A, d_rowIdx, d_colIdx);
    cusparseXcoo2csr(handle, d_rowIdx, nz, N, d_rowIdx, CUSPARSE_INDEX_BASE_ZERO);
    cusolverSpDcsrlsvchol(cusolverHandle, N, nz, descrA, d_A, d_rowIdx, d_colIdx, d_b, 0, 0, d_x, &singularity);
}

template<>
void CholeskySpImmedSolver<float>::Solve(int N, float* d_b, float* d_x, float* A, int nz, int* rowIdx, int* colIdx, float* d_guess)
{
    int singularity;
    sort_coo(N, nz, A, rowIdx, colIdx, d_A, d_rowIdx, d_colIdx);
    cusparseXcoo2csr(handle, d_rowIdx, nz, N, d_rowIdx, CUSPARSE_INDEX_BASE_ZERO);
    cusolverSpScsrlsvchol(cusolverHandle, N, nz, descrA, d_A, d_rowIdx, d_colIdx, d_b, 0, 0, d_x, &singularity);
}

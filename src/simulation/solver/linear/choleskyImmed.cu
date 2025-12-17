#include <linear/choleskyImmed.h>

template<typename T>
inline CholeskySpImmedSolver<T>::CholeskySpImmedSolver(int N)
{
    cusparseCreate(&handle);
    cusolverSpCreate(&cusolverHandle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cudaMalloc((void**)&d_rowPtrA, sizeof(int) * (N + 1));
}

template<typename T>
CholeskySpImmedSolver<T>::~CholeskySpImmedSolver()
{
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(cusolverHandle);
    cudaFree(d_rowPtrA);
    if (LinearSolver<T>::d_A) cudaFree(LinearSolver<T>::d_A);
    if (LinearSolver<T>::d_rowIdx) cudaFree(LinearSolver<T>::d_rowIdx);
    if (LinearSolver<T>::d_colIdx) cudaFree(LinearSolver<T>::d_colIdx);
}

template class CholeskySpImmedSolver<float>;
template class CholeskySpImmedSolver<double>;

template<>
void CholeskySpImmedSolver<double>::Solve(int N, double* d_b, double* d_x, double* A, int nz, int* rowIdx, int* colIdx, double* d_guess)
{
    int singularity;
    sort_coo(N, nz, A, rowIdx, colIdx, d_A, d_rowIdx, d_colIdx);
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, d_rowIdx, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSOLVER(cusolverSpDcsrlsvchol(cusolverHandle, N, nz, descrA, d_A, d_rowPtrA, d_colIdx, d_b, 0, 0, d_x, &singularity));
}

template<>
void CholeskySpImmedSolver<float>::Solve(int N, float* d_b, float* d_x, float* A, int nz, int* rowIdx, int* colIdx, float* d_guess)
{
    int singularity;
    sort_coo(N, nz, A, rowIdx, colIdx, d_A, d_rowIdx, d_colIdx);
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, d_rowIdx, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSOLVER(cusolverSpScsrlsvchol(cusolverHandle, N, nz, descrA, d_A, d_rowPtrA, d_colIdx, d_b, 0, 0, d_x, &singularity));
}

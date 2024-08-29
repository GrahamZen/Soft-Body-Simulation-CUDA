#pragma once

#include <linear/linear.h>
#include <cusolverSp.h>


template<typename T>
class CholeskySpImmedSolver : public LinearSolver<T> {
public:
    CholeskySpImmedSolver(int N);
    virtual ~CholeskySpImmedSolver() override;
    virtual void Solve(int N, T* d_b, T* d_x, T* d_A = nullptr, int nz = 0, int* d_rowIdx = nullptr, int* d_colIdx = nullptr, T* d_guess = nullptr) override;
private:
    cusparseHandle_t handle;
    cusolverSpHandle_t cusolverHandle;
    cusparseMatDescr_t descrA;
};

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
}
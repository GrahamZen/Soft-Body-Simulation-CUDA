#pragma once

#include <linear/linear.h>
#include <cusolverDn.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverSp.h>
#include <cusparse.h>


template<typename T>
class CholeskySpLinearSolver : public LinearSolver<T> {
public:
    CholeskySpLinearSolver(int threadsPerBlock, int* rowIdx, int* colIdx, T* val, int ASize, int len);
    virtual ~CholeskySpLinearSolver() override;
    virtual void Solve(int N, T* d_b, T* d_x, T* d_A = nullptr, int nz = 0, int* d_rowIdx = nullptr, int* d_colIdx = nullptr, T* d_guess = nullptr) override;
private:
    using LinearSolver<T>::d_A;
    using LinearSolver<T>::d_rowIdx;
    using LinearSolver<T>::d_colIdx;
    using LinearSolver<T>::d_rowPtrA;
    using LinearSolver<T>::capacity;

    void ComputeAMD(cusolverSpHandle_t handle, int rowsA, int nnzA, int* dev_csrRowPtrA, int* dev_csrColIndA, T* dev_csrValA);
    cusolverSpHandle_t cusolverHandle;
    cusparseMatDescr_t descrA;
    csrcholInfo_t d_info;
    void* buffer_gpu = nullptr;
    int* d_p = nullptr;
    T* dev_b_permuted = nullptr, * dev_x_permuted = nullptr;
    int n;
};

template<typename T>
class CholeskyDnLinearSolver : public LinearSolver<T> {
public:
    CholeskyDnLinearSolver(int threadsPerBlock, int* AIdx, T* AVal, int ASize, int len);
    virtual ~CholeskyDnLinearSolver() override;
    virtual void Solve(int N, T* d_b, T* d_x, T* d_A = nullptr, int nz = 0, int* d_rowIdx = nullptr, int* d_colIdx = nullptr, T* d_guess = nullptr) override;
private:
    using LinearSolver<T>::dType;

    cusolverDnParams_t params;
    int* d_info = nullptr;    /* error info */
    cusolverDnHandle_t cusolverHandle;
    void* d_work = nullptr;              /* device workspace */
    T* d_predecomposedA;
};
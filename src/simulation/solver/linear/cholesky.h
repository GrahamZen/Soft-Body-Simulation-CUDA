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
    virtual ~CholeskySpLinearSolver() override
    {
        if (d_info) { cusolverSpDestroyCsrcholInfo(d_info); d_info = nullptr; }
        if (descrA) { cusparseDestroyMatDescr(descrA); descrA = nullptr; }
        if (cusolverHandle) { cusolverSpDestroy(cusolverHandle); cusolverHandle = nullptr; }

        if (d_p) { cudaFree(d_p); d_p = nullptr; }
        if (buffer_gpu) { cudaFree(buffer_gpu); buffer_gpu = nullptr; }
        if (dev_x_permuted) { cudaFree(dev_x_permuted); dev_x_permuted = nullptr; }
        if (dev_b_permuted) { cudaFree(dev_b_permuted); dev_b_permuted = nullptr; }
    }

    virtual void Solve(int N, T* d_b, T* d_x, T* d_A = nullptr, int nz = 0, int* d_rowIdx = nullptr, int* d_colIdx = nullptr, T* d_guess = nullptr) override;
private:
    using LinearSolver<T>::d_A;
    using LinearSolver<T>::d_rowIdx;
    using LinearSolver<T>::d_colIdx;
    using LinearSolver<T>::d_rowPtrA;
    using LinearSolver<T>::capacity;

    void ComputeAMD(cusolverSpHandle_t handle, int rowsA, int nnzA, int* dev_csrRowPtrA, int* dev_csrColIndA, T* dev_csrValA);
    cusolverSpHandle_t cusolverHandle = nullptr;
    cusparseMatDescr_t descrA = nullptr;
    csrcholInfo_t d_info = nullptr;
    void* buffer_gpu = nullptr;
    int* d_p = nullptr;
    T* dev_b_permuted = nullptr;
    T* dev_x_permuted = nullptr;
    int n = 0;
};

template<typename T>
class CholeskyDnLinearSolver : public LinearSolver<T> {
public:
    CholeskyDnLinearSolver(int threadsPerBlock, int* AIdx, T* AVal, int ASize, int len);
    virtual ~CholeskyDnLinearSolver() override
    {
        if (d_info) { cudaFree(d_info); d_info = nullptr; }
        if (d_predecomposedA) { cudaFree(d_predecomposedA); d_predecomposedA = nullptr; }
        if (d_work) { cudaFree(d_work); d_work = nullptr; }
        if (cusolverHandle) { cusolverDnDestroy(cusolverHandle); cusolverHandle = nullptr; }
    }
    virtual void Solve(int N, T* d_b, T* d_x, T* d_A = nullptr, int nz = 0, int* d_rowIdx = nullptr, int* d_colIdx = nullptr, T* d_guess = nullptr) override;
private:
    using LinearSolver<T>::dType;

    cusolverDnParams_t params;
    int* d_info = nullptr;    /* error info */
    cusolverDnHandle_t cusolverHandle = nullptr;
    void* d_work = nullptr;              /* device workspace */
    T* d_predecomposedA = nullptr;
};
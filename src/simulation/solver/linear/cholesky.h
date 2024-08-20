#pragma once

#include <linear/linear.h>
#include <cusolverDn.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverSp.h>
#include <cusparse.h>


class CholeskySpLinearSolver : public LinearSolver {
public:
    CholeskySpLinearSolver(int threadsPerBlock, int* AIdx, float* val, int ASize, int len);
    virtual ~CholeskySpLinearSolver() override;
    virtual void Solve(int N, float *d_b, float *d_x, float *d_A = nullptr, int nz = 0, int *d_rowIdx = nullptr, int *d_colIdx = nullptr, float *d_guess = nullptr) override;
private:
    cusolverSpHandle_t cusolverHandle;
    cusparseMatDescr_t descrA;
    csrcholInfo_t d_info;
    void* buffer_gpu = nullptr;
    int* d_rowPtr;
    int* d_colIdx;
    float* d_val;
    int* d_P;
    float* d_tmp;
    int* d_mapBfromC;
    int n;
    int nnz;
};

class CholeskyDnLinearSolver : public LinearSolver {
public:
    CholeskyDnLinearSolver(int threadsPerBlock, int* AIdx, float* tmpVal, int ASize, int len);
    virtual ~CholeskyDnLinearSolver() override;
    virtual void Solve(int N, float *d_b, float *d_x, float *d_A = nullptr, int nz = 0, int *d_rowIdx = nullptr, int *d_colIdx = nullptr, float *d_guess = nullptr) override;
private:
    cusolverDnParams_t params;
    int* d_info = nullptr;    /* error info */
    cusolverDnHandle_t cusolverHandle;
    void* d_work = nullptr;              /* device workspace */
    float* d_predecomposedA;
};
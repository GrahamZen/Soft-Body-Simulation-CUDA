#pragma once

#include <cusolverDn.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverSp.h>
#include <cusparse.h>

class LinearSolver {
public:
    LinearSolver() = default;
    virtual ~LinearSolver() = default;
    virtual void Solve(float* d_b, int bSize, float* d_x) = 0;
};

class CholeskySplinearSolver : public LinearSolver {
public:
    CholeskySplinearSolver(int threadsPerBlock, int* AIdx, float* val, int ASize, int nnz);
    virtual ~CholeskySplinearSolver() override;
    virtual void Solve(float* d_b, int bSize, float* d_x) override;
private:
    void ApproximateMinimumDegree(int ASize, int* ARow, int* ACol, float* AVal);
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

class CholeskyDnlinearSolver : public LinearSolver {
public:
    CholeskyDnlinearSolver(int threadsPerBlock, int* AIdx, float* tmpVal, int ASize, int len);
    virtual ~CholeskyDnlinearSolver() override;
    virtual void Solve(float* d_b, int bSize, float* d_x) override;
private:
    cusolverDnParams_t params;
    int* d_info = nullptr;    /* error info */
    cusolverDnHandle_t cusolverHandle;
    void* d_work = nullptr;              /* device workspace */
    float* d_A;
};
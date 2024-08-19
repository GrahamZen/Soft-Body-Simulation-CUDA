#pragma once

#include <simulation/solver/femSolver.h>
#include <context.h>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <cusolverDn.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

class SimulationCUDAContext;

class CholeskySpData {
public:
    CholeskySpData(int threadsPerBlock, int* AIdx, float* val, int ASize, int nnz);
    ~CholeskySpData();
    void Solve(float* d_b, int bSize, float* d_x);
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

class CholeskyDnData {
public:
    CholeskyDnData(int threadsPerBlock, int* AIdx, float* tmpVal, int ASize, int len);
    ~CholeskyDnData();
    void Solve(float* d_b, int bSize, float* d_x);
private:
    cusolverDnParams_t params;
    int* d_info = nullptr;    /* error info */
    cusolverDnHandle_t cusolverHandle;
    void* d_work = nullptr;              /* device workspace */
    float* d_A;
};

class PdSolver : public FEMSolver {
public:
    PdSolver(int threadsPerBlock, const SolverData&);
    ~PdSolver();
    void SetGlobalSolver(bool useEigen) { this->useEigen = useEigen; }
    virtual void Update(SolverData& solverData, SolverParams& solverParams) override;
protected:
    virtual void SolverPrepare(SolverData& solverData, SolverParams& solverParams) override;
    virtual void SolverStep(SolverData& solverData, SolverParams& solverParams) override;
private:
    CholeskySpData* cholSpData = nullptr;
    CholeskyDnData* cholDnData = nullptr;
    bool useEigen = false;
    float* Mass;

    float* masses;
    float* sn;
    float* b;
    float* bHost;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky_decomposition_;
};

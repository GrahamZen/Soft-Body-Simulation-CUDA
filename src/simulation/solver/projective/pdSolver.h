#pragma once

#include <simulation/solver/femSolver.h>
#include <context.h>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <cusolverDn.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

class SimulationCUDAContext;
class PdSolver : public FEMSolver {
public:
    PdSolver(SimulationCUDAContext*, const SolverData&);
    ~PdSolver();
    virtual void Update(SolverData& solverData, SolverAttribute& solverAttr) override;
protected:
    virtual void SolverPrepare(SolverData& solverData, SolverAttribute& solverAttr) override;
    virtual void SolverStep(SolverData& solverData, SolverAttribute& solverAttr) override;
private:
    int nnzNumber;

    bool solverReady = false;

    float* Mass;

    cusolverDnParams_t params;
    int* d_info = nullptr;    /* error info */
    cusolverDnHandle_t cusolverHandle;
    void* d_work = nullptr;              /* device workspace */

    float* d_A;
    float* masses;
    float* sn;
    float* b;
    float* bHost;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky_decomposition_;
};

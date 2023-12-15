#pragma once
#include <simulation/solver/femSolver.h>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <thrust/device_vector.h>
#include <context.h>

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

    csrcholInfo_t d_info = nullptr;
    void* buffer_gpu = nullptr;
    cusolverSpHandle_t cusolverHandle;

    float* masses;
    float* sn;
    float* b;
    float* bHost;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky_decomposition_;
};

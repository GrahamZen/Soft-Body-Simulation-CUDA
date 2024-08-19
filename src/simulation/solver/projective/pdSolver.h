#pragma once

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <simulation/solver/femSolver.h>
#include <context.h>
#include <Eigen/Dense>

class SimulationCUDAContext;
class LinearSolver;

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
    LinearSolver* ls = nullptr;
    bool useEigen = false;
    float* Mass;

    float* masses;
    float* sn;
    float* b;
    float* bHost;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky_decomposition_;
};

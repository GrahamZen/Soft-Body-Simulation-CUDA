#pragma once

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <simulation/solver/femSolver.h>
#include <Eigen/Dense>

template<typename T>
class LinearSolver;

class PdSolver : public FEMSolver<float> {
public:
    PdSolver(int threadsPerBlock, const SolverData<float>&);
    ~PdSolver();
    void SetGlobalSolver(bool useEigen) { this->useEigen = useEigen; }
    virtual void Update(SolverData<float>& solverData, const SolverParams<float>& solverParams) override;
    virtual void Reset() override;
protected:
    virtual void SolverPrepare(SolverData<float>& solverData, const SolverParams<float>& solverParams) override;
    virtual bool SolverStep(SolverData<float>& solverData, const SolverParams<float>& solverParams) override;
private:
    LinearSolver<float>* ls = nullptr;
    bool useEigen = false;

    float* masses;
    float* sn;
    float* b;
    float* bHost;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky_decomposition_;
};

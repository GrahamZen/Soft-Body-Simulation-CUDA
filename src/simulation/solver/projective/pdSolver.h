#pragma once

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <simulation/solver/femSolver.h>
#include <Eigen/Dense>

template<typename T>
class LinearSolver;

class PdSolver : public FEMSolver<float> {
public:
    enum class SolverType
    {
        EigenCholesky, CuSolverCholesky, Jacobi
    };
    PdSolver(int threadsPerBlock, const SolverData<float>&);
    ~PdSolver();
    void SetGlobalSolver(SolverType val) { this->solverType = val; }
    virtual void Update(SolverData<float>& solverData, const SolverParams<float>& solverParams) override;
    virtual void Reset() override;
protected:
    virtual void SolverPrepare(SolverData<float>& solverData, const SolverParams<float>& solverParams) override;
    virtual bool SolverStep(SolverData<float>& solverData, const SolverParams<float>& solverParams) override;
private:
    LinearSolver<float>* ls = nullptr;
    LinearSolver<float>* jacobiSolver = nullptr;
    SolverType solverType = SolverType::CuSolverCholesky;

    float* masses;
    float* sn;
    float* sn_prime;
    float* b;
    float* bHost;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky_decomposition_;

    // jacobi
    int* AColIdx = nullptr;
    int* ARowIdx = nullptr;
    float* AVal = nullptr;
    int nnz = 0;
};

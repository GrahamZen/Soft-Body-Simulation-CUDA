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
        Jacobi, CuSolverCholesky, EigenCholesky
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
    SolverType solverType;

    const float positional_weight = 1e6;
    float* massDt_2s;
    float* sn;
    float* sn_old;
    float* b;
    float* bHost;
    float* matrix_diag;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky_decomposition_;

    //Jacobi
    float omega;
    float* next_x;
    float* prev_x;
};

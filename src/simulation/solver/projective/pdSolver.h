#pragma once

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <simulation/solver/femSolver.h>
#include <Eigen/Dense>
#include <memory>

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
    std::unique_ptr<LinearSolver<float>> ls;
    std::unique_ptr<LinearSolver<float>> jacobiSolver;
    SolverType solverType;

    const float positional_weight = 1e6;
    float* massDt_2s = nullptr;
    float* sn = nullptr;
    float* sn_old = nullptr;
    float* b = nullptr;
    float* bHost = nullptr;
    float* matrix_diag = nullptr;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky_decomposition_;

    //Jacobi
    float omega;
    float* next_x = nullptr;
    float* prev_x = nullptr;
};

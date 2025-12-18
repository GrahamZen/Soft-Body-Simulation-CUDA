#pragma once

#include <simulation/solver/femSolver.h>
#include <energy/ip.h>

template<typename T>
class LinearSolver;

class IPCSolver : public FEMSolver<double> {
public:
    IPCSolver(int threadsPerBlock, const SolverData<double>&);
    ~IPCSolver();
    virtual void Update(SolverData<double>& solverData, const SolverParams<double>& solverParams) override;
    virtual void Reset() override;
    bool EndCondition(double h, double tolerance);
protected:
    virtual void SolverPrepare(SolverData<double>& solverData, const SolverParams<double>& solverParams) override;
    virtual bool SolverStep(SolverData<double>& solverData, const SolverParams<double>& solverParams) override;
    bool SearchDirection(SolverData<double>& solverData, const SolverParams<double>& solverParams, double h2);
    void DOFElimination(SolverData<double>& solverData);
private:
    bool failed = false;
    bool* d_isFixed;
    int numVerts = 0;
    // Hessian(sparse)
    double* p = nullptr; // search direction
    glm::dvec3* xTmp = nullptr;
    glm::dvec3* x_n = nullptr;
    IPEnergy energy;
    LinearSolver<double>* linearSolver = nullptr;
};
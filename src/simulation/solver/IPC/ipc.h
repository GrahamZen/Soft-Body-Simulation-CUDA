#pragma once

#include <simulation/solver/femSolver.h>
#include <energy/ip.h>

template<typename T>
class LinearSolver;

class IPCSolver : public FEMSolver<double> {
public:
    IPCSolver(int threadsPerBlock, const SolverData<double>&, double dhat, double tolerance = 1e-2);
    ~IPCSolver();
    virtual void Update(SolverData<double>& solverData, SolverParams<double>& solverParams) override;
    bool EndCondition(double h);
protected:
    virtual void SolverPrepare(SolverData<double>& solverData, SolverParams<double>& solverParams) override;
    virtual void SolverStep(SolverData<double>& solverData, SolverParams<double>& solverParams) override;
    void SearchDirection(SolverData<double>& solverData, double h2);
    void DOFElimination(SolverData<double>& solverData);
private:
    int numVerts = 0;
    double tolerance;
    // Hessian(sparse)
    double* p = nullptr; // search direction
    glm::dvec3* xTmp = nullptr;
    glm::dvec3* x_n = nullptr;
    IPEnergy energy;
    LinearSolver<double>* linearSolver = nullptr;
};
#pragma once

#include <simulation/solver/femSolver.h>
#include <energy/corotated.h>
#include <energy/gravity.h>
#include <energy/inertia.h>

template<typename T>
class LinearSolver;

struct IPEnergy {
    IPEnergy(const SolverData<double>& solverData);
    ~IPEnergy();
    double Val(const glm::dvec3* Xs, const SolverData<double>& solverData, double h2) const;
    void Gradient(const SolverData<double>& solverData, double h2) const;
    void Hessian(const SolverData<double>& solverData, double h2) const;
    double* gradient = nullptr;
    int nnz = 0;
    double* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
    InertiaEnergy<double> inertia;
    GravityEnergy<double> gravity;
    ElasticEnergy<double>* elastic = nullptr;
};

class IPCSolver : public FEMSolver<double> {
public:
    IPCSolver(int threadsPerBlock, const SolverData<double>&);
    ~IPCSolver();
    virtual void Update(SolverData<double>& solverData, SolverParams& solverParams) override;
    bool EndCondition(double h);
protected:
    virtual void SolverPrepare(SolverData<double>& solverData, SolverParams& solverParams) override;
    virtual void SolverStep(SolverData<double>& solverData, SolverParams& solverParams) override;
    void SearchDirection(SolverData<double>& solverData, double h2);
private:
    int numVerts = 0;
    double tolerance = 1e-2;
    // Hessian(sparse)
    double* p = nullptr; // search direction
    glm::dvec3* xTmp = nullptr;
    glm::dvec3* x_n = nullptr;
    IPEnergy energy;
    LinearSolver<double>* linearSolver = nullptr;
};
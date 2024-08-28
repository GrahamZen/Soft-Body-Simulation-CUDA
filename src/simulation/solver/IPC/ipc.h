#pragma once

#include <simulation/solver/femSolver.h>
#include <energy/corotated.h>
#include <energy/gravity.h>
#include <energy/inertia.h>

class IPCSolver : public FEMSolver<double> {
public:
    IPCSolver(int threadsPerBlock, const SolverData<double>&);
    ~IPCSolver();
    virtual void Update(SolverData<double>& solverData, SolverParams& solverParams) override;
protected:
    virtual void SolverPrepare(SolverData<double>& solverData, SolverParams& solverParams) override;
    virtual void SolverStep(SolverData<double>& solverData, SolverParams& solverParams) override;
private:
    int nnz = 0;
    double totalEnergy = 0;
    double* gradient = nullptr;
    // Hessian(sparse)
    double* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
    InertiaEnergy<double> inertia;
    GravityEnergy<double> gravity;
    ElasticEnergy<double>* elastic = nullptr;
};
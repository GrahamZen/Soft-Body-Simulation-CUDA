#pragma once

#include <simulation/solver/femSolver.h>

class IPCSolver : public FEMSolver {
public:
    IPCSolver(int threadsPerBlock, const SolverData&);
    ~IPCSolver();
    virtual void Update(SolverData& solverData, SolverParams& solverParams) override;
protected:
    virtual void SolverPrepare(SolverData& solverData, SolverParams& solverParams) override;
    virtual void SolverStep(SolverData& solverData, SolverParams& solverParams) override;
private:
    int numVerts = 0;
    double totalEnergy = 0;
    double* gradient = nullptr;
    // Hessian(sparse)
    double* hessianVal = nullptr;
    int* hessianRowIdx = nullptr;
    int* hessianColIdx = nullptr;
};
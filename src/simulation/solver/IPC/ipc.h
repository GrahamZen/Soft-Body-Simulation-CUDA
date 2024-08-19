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
};
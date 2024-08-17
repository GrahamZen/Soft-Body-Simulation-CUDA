#pragma once

#include <simulation/solver/solver.h>

class SolverData;

class FEMSolver : public Solver {
public:
    FEMSolver();
    virtual ~FEMSolver() = default;

    virtual void Update(SolverData& solverData, SolverParams& solverParams) = 0;
protected:
    virtual void SolverPrepare(SolverData& solverData, SolverParams& solverParams) = 0;
    virtual void SolverStep(SolverData& solverData, SolverParams& solverParams) = 0;
};

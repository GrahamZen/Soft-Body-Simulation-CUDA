#pragma once

#include <solver.cuh>

class SolverData;

class FEMSolver : public Solver {
public:
    FEMSolver(SimulationCUDAContext*);
    virtual ~FEMSolver() = default;

    virtual void Update(SolverData& solverData, SolverAttribute& solverAttr) = 0;
protected:
    virtual void SolverPrepare(SolverData& solverData, SolverAttribute& solverAttr) = 0;
    virtual void SolverStep(SolverData& solverData, SolverAttribute& solverAttr) = 0;
};

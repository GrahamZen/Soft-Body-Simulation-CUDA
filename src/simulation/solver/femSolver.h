#pragma once

#include <solver.cuh>

class SolverData;

class FEMSolver : public Solver {
public:
    FEMSolver(SimulationCUDAContext*, SolverAttribute&);
    virtual ~FEMSolver() = default;

    virtual void Update(SolverData& solverData) = 0;
protected:
    virtual void SolverPrepare(SolverData& solverData) = 0;
    virtual void SolverStep(SolverData& solverData) = 0;
};

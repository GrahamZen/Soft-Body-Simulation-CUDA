#pragma once

#include <simulation/solver/solver.h>

template<typename HighP>
class FEMSolver : public Solver<HighP> {
public:
    FEMSolver(int threadsPerBlock);
    virtual ~FEMSolver() = default;

    virtual void Update(SolverData<HighP>& solverData, SolverParams& solverParams) = 0;
protected:
    virtual void SolverPrepare(SolverData<HighP>& solverData, SolverParams& solverParams) = 0;
    virtual void SolverStep(SolverData<HighP>& solverData, SolverParams& solverParams) = 0;
};

template<typename HighP>
FEMSolver<HighP>::FEMSolver(int threadsPerBlock) : Solver<HighP>(threadsPerBlock) {
}
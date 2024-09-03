#pragma once

#include <simulation/solver/solver.h>

template<typename Scalar>
class FEMSolver : public Solver<Scalar> {
public:
    FEMSolver(int threadsPerBlock, const SolverData<Scalar>& solverData);
    virtual ~FEMSolver() = default;

    virtual void Update(SolverData<Scalar>& solverData, SolverParams<Scalar>& solverParams) = 0;
protected:
    virtual void SolverPrepare(SolverData<Scalar>& solverData, SolverParams<Scalar>& solverParams) = 0;
    virtual void SolverStep(SolverData<Scalar>& solverData, SolverParams<Scalar>& solverParams) = 0;
};
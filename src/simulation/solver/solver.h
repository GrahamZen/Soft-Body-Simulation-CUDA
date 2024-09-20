#pragma once

#include <def.h>

struct SoftBodyData {
    indexType* Tri = nullptr;
    int numTris = 0;
};

template<typename Scalar>
class Solver {
public:
    Solver(int threadsPerBlock);
    virtual ~Solver();

    virtual void Update(SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) = 0;
    virtual void Reset();
protected:
    virtual void SolverPrepare(SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) = 0;
    virtual bool SolverStep(SolverData<Scalar>& solverData, const SolverParams<Scalar>& solverParams) = 0;
    int threadsPerBlock;

    bool solverReady = false;
};

template<typename Scalar>
Solver<Scalar>::Solver(int threadsPerBlock) : threadsPerBlock(threadsPerBlock)
{
}

template<typename Scalar>
Solver<Scalar>::~Solver() {
}

template<typename Scalar>
inline void Solver<Scalar>::Reset()
{
    solverReady = false;
}